"""Finite policy rollouts with immediate slot refilling and terminal masking."""

import time

import jax
import jax.numpy as jnp
import numpy as np


def make_policy_rollout_runner(
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
    execution="host",
    chunk_steps=256,
):
    """Run each seeded episode once; retain only its pre-terminal transitions.

    Policy inference stays outside the Warp simulation executable. A completed
    slot gets the next episode's explicit seed and zero action history. Counts,
    returns and trajectories are indexed by episode, independent of slot order.
    """
    if execution == "compiled":
        from learning.reinforcement.compiled_rollouts import (
            make_compiled_rollout_runner,
        )

        return make_compiled_rollout_runner(
            environment,
            policy_action,
            policy_observation,
            episodes=episodes,
            batch_size=batch_size,
            episode_length=episode_length,
            action_size=action_size,
            capture_successes=capture_successes,
            progress_callback=progress_callback,
            chunk_steps=chunk_steps,
        )
    if execution != "host":
        raise ValueError("Rollout execution must be host or compiled")
    if min(episodes, batch_size, episode_length) < 1:
        raise ValueError("Rollout counts, batch size and horizon must be positive")
    batch_size = min(batch_size, episodes)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)

    @jax.jit
    def replace_slots(previous, fresh, mask):
        def select(old, new):
            if hasattr(old, "shape") and old.ndim and old.shape[0] == batch_size:
                expanded = mask.reshape((batch_size,) + (1,) * (old.ndim - 1))
                return jnp.where(expanded, new, old)
            return old

        return jax.tree_util.tree_map(select, previous, fresh)

    def run(state, *, seed_start):
        started = time.perf_counter()
        slot_ids = np.arange(batch_size)
        next_episode = batch_size
        env_state = jax.block_until_ready(
            reset(jax.vmap(jax.random.PRNGKey)(jnp.asarray(seed_start + slot_ids)))
        )
        ema = jnp.zeros((batch_size, action_size), dtype=jnp.float32)
        valid = jnp.zeros((batch_size, 1), dtype=bool)
        lengths = np.zeros(episodes, dtype=np.int32)
        returns = np.zeros(episodes, dtype=np.float64)
        successes = np.zeros(episodes, dtype=bool)
        # Each active slot stores at most one unfinished episode. Failed paths
        # are discarded immediately; successful paths are ordered by seed below.
        paths = [[] for _ in range(batch_size)] if capture_successes else None
        completed_paths = {}
        completed = 0
        vector_steps = 0

        def report(event):
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": event,
                        "episodes": completed,
                        "total_episodes": episodes,
                        "batch_size": batch_size,
                        "steps": vector_steps,
                        "active_episodes": int(np.count_nonzero(slot_ids >= 0)),
                        "successes": int(successes.sum()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )

        report("rollouts_started")
        while completed < episodes:
            action = jax.block_until_ready(
                policy_action(
                    state.normalizer_state,
                    state.policy_params,
                    env_state.obs,
                    ema,
                    valid,
                    jax.random.fold_in(jax.random.PRNGKey(seed_start), vector_steps),
                )
            )
            observation = (
                np.asarray(policy_observation(env_state.obs, ema))
                if capture_successes
                else None
            )
            next_state = jax.block_until_ready(step(env_state, action))
            action = next_state.info.get("executed_normalized_action", action)
            reward, done = jax.device_get((next_state.reward, next_state.done))
            active = slot_ids >= 0
            episode_ids = slot_ids[active]
            lengths[episode_ids] += 1
            returns[episode_ids] += np.asarray(reward)[active]
            terminal = active & (
                np.asarray(done).astype(bool)
                | (lengths[np.maximum(slot_ids, 0)] >= episode_length)
            )
            # Preserve the existing replay convention: previous action is zero
            # in the terminal next-observation, whose discount will also be zero.
            next_ema = jnp.where(jnp.asarray(terminal)[:, None], 0.0, action)
            next_valid = ~jnp.asarray(terminal)[:, None]
            if capture_successes:
                next_observation = np.asarray(
                    policy_observation(next_state.obs, next_ema)
                )
                host_action = np.asarray(action)
                for slot in np.flatnonzero(active):
                    paths[slot].append(
                        (
                            observation[slot].copy(),
                            host_action[slot].copy(),
                            next_observation[slot].copy(),
                            float(reward[slot]),
                        )
                    )
            refill = np.zeros(batch_size, dtype=bool)
            for slot in np.flatnonzero(terminal):
                episode = int(slot_ids[slot])
                # A timeout with positive step reward is not a success: require
                # the environment's actual terminal signal as well.
                success = bool(done[slot]) and reward[slot] > 0.0
                successes[episode] = success
                if capture_successes:
                    if success:
                        completed_paths[episode] = paths[slot]
                    paths[slot] = []
                completed += 1
                if next_episode < episodes:
                    slot_ids[slot] = next_episode
                    next_episode += 1
                    refill[slot] = True
                else:
                    slot_ids[slot] = -1
            if np.any(refill):
                # Reset a fresh vector and select only refilled slots, leaving
                # in-flight episodes, their RNG state and DR factors untouched.
                keys = jax.vmap(jax.random.PRNGKey)(
                    jnp.asarray(seed_start + np.maximum(slot_ids, 0))
                )
                fresh = jax.block_until_ready(reset(keys))
                next_state = jax.block_until_ready(
                    replace_slots(next_state, fresh, jnp.asarray(refill))
                )
            env_state, ema, valid = next_state, next_ema, next_valid
            vector_steps += 1
            if vector_steps % 100 == 0:
                report("steps_completed")
        candidates = []
        for episode in sorted(completed_paths):
            path = completed_paths[episode]
            observations, actions, next_observations, rewards = zip(*path, strict=True)
            candidates.append(
                (
                    len(path),
                    np.asarray(observations),
                    np.asarray(actions),
                    np.asarray(next_observations),
                    np.asarray(rewards),
                )
            )
        successful_lengths = lengths[successes]
        summary = {
            "episodes": episodes,
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "mean_length": float(lengths.mean()),
            "mean_reward": float(returns.mean()),
            "vector_steps": vector_steps,
            "wall_seconds": time.perf_counter() - started,
            "successful_mean_length": (
                float(successful_lengths.mean())
                if successes.any()
                else float(episode_length)
            ),
        }
        report("rollouts_completed")
        return summary, candidates

    return run
