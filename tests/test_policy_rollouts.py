"""Seeded rolling slots preserve trajectories and avoid fixed-horizon work."""

from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.envs.base import State

from learning.reinforcement.policy_rollouts import make_policy_rollout_runner


class SeededEnvironment:
    def reset(self, keys):
        identifiers = keys[:, 1].astype(jp.int32)
        n = len(keys)
        return State(
            pipeline_state=None,
            obs=jp.stack((identifiers, jp.zeros(n)), axis=1),
            reward=jp.zeros(n),
            done=jp.zeros(n),
            metrics={},
            info={"id": identifiers, "t": jp.zeros(n)},
        )

    def step(self, state, action):
        t = state.info["t"] + 1
        identifiers = state.info["id"]
        limit = jp.where(identifiers == 0, 20, 2)
        done = t >= limit
        reward = jp.where(done, jp.where(identifiers == 0, -1.0, 1.0), -0.01)
        return state.replace(
            obs=jp.stack((identifiers, t), axis=1),
            reward=reward,
            done=done.astype(float),
            info={"id": identifiers, "t": t},
        )


@jax.jit
def policy(normalizer, params, obs, previous, valid, key):
    # Echo history into the action; initial valid=False must produce zero.
    return jp.where(valid, previous + 0.01, jp.zeros((len(obs), 6)))


def observation(obs, previous):
    return jp.concatenate((obs, previous), axis=1)


@pytest.mark.parametrize("execution", ["host", "compiled"])
def test_refill_preserves_seed_identity_history_and_first_terminal_transition(
    execution,
):
    progress = []
    run = make_policy_rollout_runner(
        SeededEnvironment(),
        policy,
        observation,
        episodes=6,
        batch_size=2,
        episode_length=30,
        action_size=6,
        capture_successes=True,
        execution=execution,
        chunk_steps=3,
        progress_callback=progress.append,
    )
    report, paths = run(
        SimpleNamespace(normalizer_state=(), policy_params=()), seed_start=0
    )
    assert report["episodes"] == 6
    assert report["successes"] == 5
    assert report["mean_length"] == 5
    assert report["vector_steps"] == 20  # Fixed batches would take 20 + 2 + 2.
    assert report["mean_reward"] == pytest.approx((-1.19 + 5 * 0.99) / 6)
    assert [int(path[1][0, 0]) for path in paths] == [1, 2, 3, 4, 5]
    for length, obs, actions, next_obs, rewards in paths:
        assert length == 2
        np.testing.assert_array_equal(obs[:, 1], [0, 1])
        np.testing.assert_array_equal(next_obs[:, 1], [1, 2])
        np.testing.assert_allclose(obs[0, 2:], 0)
        np.testing.assert_allclose(actions[0], 0)
        np.testing.assert_allclose(actions[1], 0.01)
        np.testing.assert_allclose(next_obs[-1, 2:], 0)
        np.testing.assert_allclose(rewards, [-0.01, 1])
    assert progress[-1]["active_episodes"] == 0
    assert progress[-1]["episodes"] == 6


@pytest.mark.parametrize("execution", ["host", "compiled"])
def test_refill_stops_at_horizon_and_does_not_collect_timeout_as_success(execution):
    class NeverDone(SeededEnvironment):
        def step(self, state, action):
            state = super().step(state, action)
            return state.replace(
                done=jp.zeros_like(state.done), reward=jp.ones_like(state.reward)
            )

    run = make_policy_rollout_runner(
        NeverDone(),
        policy,
        observation,
        episodes=5,
        batch_size=2,
        episode_length=3,
        action_size=6,
        capture_successes=True,
        execution=execution,
        chunk_steps=4,
    )
    report, paths = run(
        SimpleNamespace(normalizer_state=(), policy_params=()), seed_start=10
    )
    assert report["episodes"] == 5
    assert report["successes"] == 0
    assert report["mean_length"] == 3
    assert report["vector_steps"] == 9
    assert not paths


@pytest.mark.parametrize("execution", ["host", "compiled"])
def test_episode_outcomes_are_independent_of_slot_count(execution):
    reports = []
    for batch_size in (1, 2, 6):
        run = make_policy_rollout_runner(
            SeededEnvironment(),
            policy,
            observation,
            episodes=6,
            batch_size=batch_size,
            episode_length=30,
            action_size=6,
            execution=execution,
            chunk_steps=4,
        )
        report, paths = run(
            SimpleNamespace(normalizer_state=(), policy_params=()), seed_start=0
        )
        assert not paths
        reports.append(report)
    for field in (
        "episodes",
        "successes",
        "success_rate",
        "mean_length",
        "mean_reward",
        "successful_mean_length",
    ):
        assert reports[0][field] == pytest.approx(reports[1][field])
        assert reports[0][field] == pytest.approx(reports[2][field])
    assert reports[0]["vector_steps"] == 30
    assert reports[2]["vector_steps"] == 20


def test_compiled_wrapper_removes_eager_reset_without_changing_episode_results():
    from brax.envs.wrappers.training import EpisodeWrapper
    from mujoco_playground._src.mjx_env import State as PlaygroundState
    from mujoco_playground._src.wrapper import BraxAutoResetWrapper

    class Environment(SeededEnvironment):
        def reset(self, keys):
            state = super().reset(keys)
            return PlaygroundState(
                data=jp.zeros((len(keys), 1)),
                obs=state.obs,
                reward=state.reward,
                done=state.done,
                metrics=state.metrics,
                info=state.info,
            )

        def step(self, state, action):
            following = super().step(state, action)
            return following.replace(info={**state.info, **following.info})

    reports = []
    for execution in ("host", "compiled"):
        env = BraxAutoResetWrapper(
            EpisodeWrapper(Environment(), 30, 1), full_reset=True
        )
        run = make_policy_rollout_runner(
            env,
            policy,
            observation,
            episodes=8,
            batch_size=2,
            episode_length=30,
            action_size=6,
            execution=execution,
            chunk_steps=3,
        )
        report, _ = run(
            SimpleNamespace(normalizer_state=(), policy_params=()), seed_start=0
        )
        reports.append(report)
    for name in ("episodes", "successes", "mean_reward", "mean_length", "vector_steps"):
        assert reports[0][name] == pytest.approx(reports[1][name])


def test_compiled_step_retains_updated_shared_simulator_state():
    class SharedState(SeededEnvironment):
        def reset(self, keys):
            state = super().reset(keys)
            return state.replace(info={**state.info, "shared": jp.asarray(0)})

        def step(self, state, action):
            following = super().step(state, action)
            shared = state.info["shared"] + 1
            return following.replace(
                obs=following.obs.at[:, 1].set(shared),
                info={**following.info, "shared": shared},
            )

    run = make_policy_rollout_runner(
        SharedState(),
        policy,
        observation,
        episodes=2,
        batch_size=2,
        episode_length=10,
        action_size=6,
        execution="compiled",
        capture_successes=True,
        chunk_steps=3,
    )
    report, paths = run(
        SimpleNamespace(normalizer_state=(), policy_params=()), seed_start=1
    )
    assert report["successes"] == 2
    for path in paths:
        np.testing.assert_array_equal(path[3][:, 1], [1, 2])
