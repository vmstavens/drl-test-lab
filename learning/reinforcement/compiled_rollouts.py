"""GPU-resident rollout chunks, including policy, termination and slot refills."""

import time
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np


def _explicit_reset_environment(environment):
    """Keep the existing seed transform, avoiding an unused full reset per step."""
    from mujoco_playground._src.wrapper import BraxAutoResetWrapper

    if not isinstance(environment, BraxAutoResetWrapper):
        return environment

    class ExplicitReset:
        reset = environment.reset

        @staticmethod
        def step(state, action):
            # The outer rollout controller owns all terminal masks and refills.
            # Return the actual terminal state; replay discount is zero there.
            return environment.env.step(
                state.replace(done=jp.zeros_like(state.done)), action
            )

    return ExplicitReset()


class _Rollouts(NamedTuple):
    env: object
    ema: object
    valid: object
    slots: object
    next_episode: object
    lengths: object
    returns: object
    successes: object
    completed: object
    steps: object
    rng: object


def make_compiled_rollout_runner(
    environment,
    policy_action,
    policy_observation,
    *,
    episodes,
    batch_size,
    episode_length,
    action_size,
    capture_successes=False,
    progress_callback=None,
    chunk_steps=256,
):
    """Only synchronize chunk summaries/trajectory blocks, never individual steps.

    Chunk boundaries provide progress and bound replay-transfer memory. All
    feedback decisions, terminal masks, seeds and resets execute inside JIT.
    """
    if min(episodes, batch_size, episode_length, chunk_steps) < 1:
        raise ValueError("Rollout counts, horizon and chunk size must be positive")
    batch_size = min(batch_size, episodes)
    environment = _explicit_reset_environment(environment)
    reset = jax.jit(environment.reset)

    def replace_slots(old, fresh, mask, *, preserve_shared=True):
        def select(a, b):
            if hasattr(a, "shape") and a.ndim and a.shape[0] == batch_size:
                return jp.where(mask.reshape((batch_size,) + (1,) * (a.ndim - 1)), b, a)
            return a if preserve_shared else b

        return jax.tree_util.tree_map(select, old, fresh)

    def advance(carry, norm, params, cache):
        active = carry.slots >= 0
        ids = jp.where(active, carry.slots, episodes)
        action = policy_action(
            norm,
            params,
            carry.env.obs,
            carry.ema,
            carry.valid,
            jax.random.fold_in(carry.rng, carry.steps),
        )
        following = environment.step(carry.env, action)
        action = following.info.get("executed_normalized_action", action)
        following = replace_slots(carry.env, following, active, preserve_shared=False)
        lengths = carry.lengths.at[ids].add(active.astype(jp.int32))
        returns = carry.returns.at[ids].add(jp.where(active, following.reward, 0.0))
        done = active & (following.done.astype(bool) | (lengths[ids] >= episode_length))
        succeeded = done & following.done.astype(bool) & (following.reward > 0.0)
        successes = carry.successes.at[ids].max(succeeded)
        ema = jp.where(done[:, None], 0.0, action)
        valid = ~done[:, None]
        trajectory = None
        if capture_successes:
            trajectory = (
                carry.slots,
                policy_observation(carry.env.obs, carry.ema),
                action,
                policy_observation(following.obs, ema),
                following.reward,
                done,
                succeeded,
            )
        ranks = jp.cumsum(done.astype(jp.int32)) - 1
        assigned = carry.next_episode + ranks
        refill = done & (assigned < episodes)
        slots = jp.where(done, jp.where(refill, assigned, -1), carry.slots)

        # Resets are sampled once, with the same explicit per-episode seeds.
        # Device-side gathers avoid host-driven conditional reset callbacks.
        reset_ids = jp.where(refill, assigned, jp.maximum(carry.slots, 0))
        reset_ids = jp.minimum(reset_ids, episodes - 1)

        def from_cache(current, pool):
            if hasattr(pool, "shape") and pool.ndim and pool.shape[0] == episodes:
                return pool[reset_ids]
            return current

        fresh = jax.tree_util.tree_map(from_cache, following, cache)
        following = replace_slots(following, fresh, done)
        state = _Rollouts(
            following,
            ema,
            valid,
            slots,
            carry.next_episode + jp.sum(refill.astype(jp.int32)),
            lengths,
            returns,
            successes,
            carry.completed + jp.sum(done.astype(jp.int32)),
            carry.steps + jp.any(active).astype(jp.int32),
            carry.rng,
        )
        return state, trajectory

    @jax.jit
    def chunk(carry, norm, params, cache):
        # Static scan: episode decisions and selections stay on device. At most
        # chunk_steps-1 parked padding steps follow the final termination.
        return jax.lax.scan(
            lambda state, _: advance(state, norm, params, cache),
            carry,
            None,
            length=chunk_steps,
        )

    def run(state, *, seed_start):
        started = time.perf_counter()
        slots = jp.arange(batch_size, dtype=jp.int32)
        # Keep reset batches within the configured world count: a single
        # 512-world reset could overflow a buffer sized for 128 worlds.
        pools, sizes = [], []
        for offset in range(0, episodes, batch_size):
            size = min(batch_size, episodes - offset)
            keys = jax.vmap(jax.random.PRNGKey)(
                seed_start + jp.arange(offset, offset + size, dtype=jp.int32)
            )
            pools.append(reset(keys))
            sizes.append(size)

        def join(*arrays):
            if all(
                hasattr(a, "shape") and a.ndim and a.shape[0] == size
                for a, size in zip(arrays, sizes, strict=True)
            ):
                return jp.concatenate(arrays, axis=0)
            return arrays[0]

        cache = jax.tree_util.tree_map(join, *pools)
        env = pools[0]
        carry = _Rollouts(
            env,
            jp.zeros((batch_size, action_size)),
            jp.zeros((batch_size, 1), dtype=bool),
            slots,
            jp.asarray(batch_size),
            jp.zeros(episodes + 1, dtype=jp.int32),
            jp.zeros(episodes + 1),
            jp.zeros(episodes + 1, dtype=bool),
            jp.asarray(0),
            jp.asarray(0),
            jax.random.PRNGKey(seed_start),
        )
        pieces, candidates = {}, {}
        completed, steps = 0, 0

        def report(event, count, successes, active):
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": event,
                        "episodes": count,
                        "total_episodes": episodes,
                        "batch_size": batch_size,
                        "steps": steps,
                        "active_episodes": active,
                        "successes": successes,
                        "elapsed_seconds": time.perf_counter() - started,
                        "execution": "compiled",
                        "chunk_steps": chunk_steps,
                    }
                )

        report("rollouts_started", 0, 0, batch_size)
        while completed < episodes:
            carry, trajectory = chunk(
                carry, state.normalizer_state, state.policy_params, cache
            )
            # Transfer only scalar summaries and, for collection, bounded blocks.
            counts, trajectory = jax.device_get(
                (
                    (
                        carry.completed,
                        carry.steps,
                        jp.sum(carry.successes),
                        jp.sum(carry.slots >= 0),
                    ),
                    trajectory,
                )
            )
            completed, steps, successes, active = map(int, counts)
            if capture_successes:
                ids, obs, actions, next_obs, rewards, terminal, success = trajectory
                for episode in np.unique(ids[ids >= 0]):
                    mask = ids == episode
                    pieces.setdefault(int(episode), []).append(
                        (obs[mask], actions[mask], next_obs[mask], rewards[mask])
                    )
                    if np.any(terminal[mask]):
                        path = pieces.pop(int(episode))
                        if np.any(success[mask]):
                            arrays = tuple(
                                np.concatenate(x) for x in zip(*path, strict=True)
                            )
                            candidates[int(episode)] = (len(arrays[1]), *arrays)
            report("steps_completed", completed, successes, active)
        lengths, returns, successes = jax.device_get(
            (
                carry.lengths[:episodes],
                carry.returns[:episodes],
                carry.successes[:episodes],
            )
        )
        summary = {
            "episodes": episodes,
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "mean_length": float(lengths.mean()),
            "mean_reward": float(returns.mean()),
            "vector_steps": steps,
            "wall_seconds": time.perf_counter() - started,
            "successful_mean_length": (
                float(lengths[successes].mean())
                if successes.any()
                else float(episode_length)
            ),
            "execution": "compiled",
            "chunk_steps": chunk_steps,
        }
        report("rollouts_completed", completed, int(successes.sum()), 0)
        return summary, [candidates[i] for i in sorted(candidates)]

    return run
