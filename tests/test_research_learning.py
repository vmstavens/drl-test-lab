"""Portable tests of the actual compiled learning path and its compatibility rules."""

import json

import jax
import jax.numpy as jp
import numpy as np
import pytest

from learning.reinforcement.drlr2 import Trainer, prefer_reference
from learning.reinforcement.recursive import evaluate_gate, require_success
from learning.reinforcement.rma import adaptation_module, student_action
from learning.reinforcement.task import Task
from tests.test_learning_infrastructure import compose_workflow


def small_config():
    config = compose_workflow("train_recursive_drlr_rma")
    config["task"]["env"]["episode_length"] = 3
    config["rl"].update(
        num_envs=2,
        batch_size=8,
        memory_size=64,
        warmup_vector_steps=2,
        actor_hidden_sizes=[8],
        critic_hidden_sizes=[8],
        encoder_hidden_sizes=[8],
        latent_dim=2,
        chunk_steps=2,
    )
    config["rl"]["evaluation_gate"].update(
        episodes=4, progress_episodes=2, batch_size=2
    )
    return config


@pytest.fixture
def learner():
    config = small_config()
    task = Task(config, domain_level=0.5)

    def reference(raw, past, previous, factors, key):
        return jp.full((len(raw), 2), 0.6)

    trainer = Trainer(config, task, reference, center=np.zeros(4), scale=np.ones(4))
    yield trainer
    task.close()


def test_twin_critic_selection_requires_both_critics_and_prefers_teacher_on_ties():
    np.testing.assert_array_equal(
        prefer_reference(
            jp.array([[2.0, 2.0], [2.0, 0.0], [1.0, 1.0]]), jp.ones((3, 2))
        ),
        [[False], [True], [True]],
    )


def test_warmup_history_replay_terminal_reset_and_frozen_actor(learner):
    state = learner.initialize()
    original = state.policy_params
    first, _ = learner._step(state)
    o, a = learner.o, learner.a
    np.testing.assert_allclose(first.replay["action"][:2], 0.6)
    np.testing.assert_array_equal(first.replay["observation"][:2, o : o + a], 0)
    np.testing.assert_allclose(first.replay["next_observation"][:2, o : o + a], 0.6)
    reference = jax.tree.map(lambda array: array[:2], first.replay)
    second, _ = learner.train_chunk(first, reference, 1)
    assert int(second.updates) == 1
    for before, after in zip(
        jax.tree.leaves(original), jax.tree.leaves(second.policy_params), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    terminal, _ = learner._step(second)
    np.testing.assert_array_equal(terminal.previous_action, 0)
    np.testing.assert_array_equal(
        terminal.replay["next_observation"][4:6, o : o + a], 0
    )
    np.testing.assert_array_equal(terminal.replay["discount"][4:6], 0)


@pytest.mark.parametrize("start_step,updates", [(1, 3), (3, 4)])
def test_fused_and_split_do_same_updates_from_same_state(learner, start_step, updates):
    state, _ = learner._step(learner.initialize())
    state = state.replace(vector_steps=jp.array(start_step))
    reference = jax.tree.map(lambda array: array[:2], state.replay)
    fused, fused_metrics = learner.train_chunk(state, reference, 2)
    learner.rl["training_execution"] = "split"
    split, split_metrics = learner.train_chunk(state, reference, 2)
    assert int(fused.updates) == int(split.updates) == updates
    assert int(fused.vector_steps) == int(split.vector_steps) == start_step + 2
    for name in fused_metrics:
        np.testing.assert_allclose(fused_metrics[name], split_metrics[name], atol=1e-5)
    for before, after in zip(
        jax.tree.leaves(fused.policy_params),
        jax.tree.leaves(split.policy_params),
        strict=True,
    ):
        np.testing.assert_allclose(before, after, atol=2e-6)
    np.testing.assert_array_equal(fused.log_alpha, state.log_alpha)
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree.leaves(fused.policy_params)
    )


def test_evaluation_never_calls_reference(learner):
    state = learner.initialize()

    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher called by evaluation")

    learner.reference = forbidden
    report, _ = learner.rollout(state.policy_params, episodes=4, seed_start=2026090900)
    assert report["episodes"] == 4
    assert report["evaluation_policy"] == "learner_only"


def test_student_actions_are_independent_of_privileged_factors(learner):
    config = compose_workflow("train_recursive_rma_phase2")["rma"]
    adaptation = adaptation_module(learner, config)
    history = jp.zeros((2, config["history_length"], learner.o + learner.a))
    params = adaptation.init(jax.random.PRNGKey(0), history)
    frozen = learner.initialize().policy_params
    first = jp.zeros((2, learner.o * 2 + learner.f))
    second = first.at[:, learner.o : learner.o + learner.f].set(1.0)
    a = student_action(
        learner, frozen, adaptation, params, first, jp.zeros((2, learner.a)), history
    )
    b = student_action(
        learner, frozen, adaptation, params, second, jp.zeros((2, learner.a)), history
    )
    np.testing.assert_array_equal(a, b)


def test_failed_promotion_and_bootstrap_cannot_be_bypassed():
    from types import SimpleNamespace

    config = small_config()
    calls = []

    def rollout(params, *, episodes, seed_start):
        calls.append((episodes, seed_start))
        return {"success_rate": 1.0 if len(calls) == 1 else 0.5}, []

    trainer = SimpleNamespace(rl=config["rl"], rollout=rollout)
    assert not evaluate_gate(trainer, SimpleNamespace(policy_params=()))["accepted"]
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    with pytest.raises(RuntimeError, match="Reference gate failed"):
        require_success(
            dict(successes=39, episodes=64, success_rate=39 / 64), count=40, rate=0.8
        )


def test_recovery_restores_exact_simulation_replay_rng_and_updates(learner, tmp_path):
    from learning.reinforcement.crash_recovery import load_recovery, save_recovery

    state, _ = learner._step(learner.initialize())
    reference = jax.tree.map(lambda array: array[:2], state.replay)
    path = tmp_path / "recovery.zip"
    save_recovery(path, state, {"next_evaluation": 20}, learner.config)
    expected, _ = learner.train_chunk(state, reference, 2)
    restored, metadata = load_recovery(path, learner.initialize(), learner.config)
    actual, _ = learner.train_chunk(restored, reference, 2)
    assert metadata["next_evaluation"] == 20
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_training_resume_preserves_evaluation_schedule_without_repeating_warmup(
    learner, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import learning.reinforcement.recursive as recursive
    from learning.reinforcement.crash_recovery import load_recovery

    learner.rl["budget"]["value"] = 8
    learner.rl["chunk_steps"] = 1
    learner.rl["checkpoint_interval_transitions"] = 2
    learner.rl["evaluation_gate"].update(minimum_transitions=4, interval_transitions=4)
    learner.reference.fingerprint = "scripted-test-reference"
    state, _ = learner._step(learner.initialize())
    reference = jax.tree.map(lambda array: array[:2], state.replay)
    np.savez(tmp_path / "reference_replay.npz", **jax.device_get(reference))
    calls = []

    def gate(trainer, state):
        calls.append(int(state.vector_steps))
        return {"accepted": len(calls) == 2, "progress": {"success_rate": 0.5}}

    monkeypatch.setattr(recursive, "evaluate_gate", gate)
    original = learner.train_chunk

    def crash(state, reference, steps):
        if int(state.vector_steps) == 2:
            raise RuntimeError("simulated interruption")
        return original(state, reference, steps)

    learner.train_chunk = crash
    writer = SimpleNamespace(add_scalar=lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        recursive.run_learning(learner, state, reference, tmp_path, writer)
    restored, saved = load_recovery(
        tmp_path / "recovery.zip", learner.initialize(), learner.config
    )
    assert int(restored.vector_steps) == 2
    learner.train_chunk = original
    restored, accepted = recursive.run_learning(
        learner,
        restored,
        reference,
        tmp_path,
        writer,
        **{
            key: saved[key]
            for key in ("accepted", "next_evaluation", "next_checkpoint")
        },
    )
    assert accepted and calls == [2, 4]
    assert int(restored.vector_steps) == 4
    assert (
        int(restored.updates) == 5
    )  # initial fixture step intentionally had no update


def test_recursive_handoff_uses_accepted_parent_and_carries_successful_replay(
    tmp_path, monkeypatch
):
    import learning.reinforcement.recursive as module

    config = small_config()
    config["il"]["checkpoint"] = "fixture-only.pkl"
    config["recursive"]["levels"] = [0.0, 0.5]
    config["rl"]["budget"]["value"] = 4
    config["rl"]["evaluation_gate"].update(
        minimum_transitions=4, interval_transitions=4
    )
    config["rl"]["handoff"].update(episodes=4, minimum_successes=1)
    width = 4 + 2 + 2
    observations = np.zeros((2, width), np.float32)
    following = observations.copy()
    actions = np.zeros((2, 2), np.float32)
    path = (2, observations, actions, following, np.array([-0.01, 99.99], np.float32))
    replay = module.replay_from_paths([path])

    def reference(raw, past, previous, factors, key):
        return jp.zeros((len(raw), 2))

    reference.center, reference.scale = jp.zeros(4), jp.ones(4)
    reference.fingerprint = "scripted-fixture"
    monkeypatch.setattr(module, "DPReference", lambda *args, **kwargs: reference)
    monkeypatch.setattr(
        module, "collect_bootstrap", lambda trainer: ({"successes": 64}, replay)
    )
    monkeypatch.setattr(
        module,
        "evaluate_gate",
        lambda trainer, state: {
            "accepted": True,
            "progress": {"success_rate": 1.0},
            "promotion": {"success_rate": 1.0},
        },
    )
    monkeypatch.setattr(
        Trainer,
        "rollout",
        lambda self, params, **kwargs: (
            {
                "success_rate": 1.0,
                "successes": kwargs["episodes"],
                "episodes": kwargs["episodes"],
                "evaluation_policy": "learner_only",
            },
            [path],
        ),
    )
    module.train_recursive(config, tmp_path)
    first = json.loads((tmp_path / "stage_000/summary.json").read_text())
    second = json.loads((tmp_path / "stage_001/summary.json").read_text())
    assert first["accepted"] and second["accepted"]
    assert (
        second["transitions"] == 0
    )  # full initial learner gate accepted the inherited actor
    with np.load(tmp_path / "stage_001/reference_replay.npz") as data:
        assert len(data["reward"]) == 4
    parent = json.loads((tmp_path / "stage_001/reference.json").read_text())[
        "parent_stage"
    ]
    assert parent == str((tmp_path / "stage_000").resolve())


def test_phase2_trains_adapter_only_and_reloads_student_with_executed_feedback(
    learner, tmp_path, monkeypatch
):
    import flax.serialization

    import learning.reinforcement.rma as module
    from learning.policies.research import load_rma_policy

    source, output = tmp_path / "phase1", tmp_path / "phase2"
    source.mkdir()
    output.mkdir()
    state = learner.initialize()
    learner.config["task"]["env"]["domain_level"] = 0.5
    learner.save_actor(source / "accepted_actor.flax", state)
    (source / "summary.json").write_text(json.dumps({"accepted": True}))
    original = (source / "accepted_actor.flax").read_bytes()
    config = compose_workflow("train_recursive_rma_phase2")
    config["phase1_job"] = str(source)
    config["rma"].update(
        probabilities=[0.0, 1.0],
        num_envs=2,
        collection_steps=2,
        updates_per_round=2,
        batch_size=4,
        replay_capacity=16,
        sample_stride=1,
        evaluation_episodes=2,
    )

    def evaluate(trainer, frozen, adaptation, params, settings):
        assert flax.serialization.to_bytes(frozen) == original
        return dict(
            episodes=2,
            success_rate=1.0,
            evaluation_policy="learner_only",
            conditioning="observable_history",
        )

    # Isolate export/gate plumbing; this unit fixture does not claim learned success.
    monkeypatch.setattr(module, "evaluate_student", evaluate)
    module.train_phase2(config, output)
    assert (source / "accepted_actor.flax").read_bytes() == original
    metadata = json.loads((output / "policy.json").read_text())
    policy = load_rma_policy(
        checkpoint=output / "adaptation.flax",
        config=metadata["config"],
        deterministic=True,
    )
    action = policy.act(np.zeros(4, np.float32))
    assert action.shape == (2,) and np.isfinite(action).all()
    policy.observe_transition(np.array([0.2, -0.1], np.float32))
    np.testing.assert_allclose(policy.previous, [0.2, -0.1], atol=1e-7)
    np.testing.assert_allclose(policy.history[-1, -2:], [0.2, -0.1], atol=1e-7)
    policy.reset(seed=7)
    np.testing.assert_array_equal(policy.previous, 0)
    np.testing.assert_array_equal(policy.history, 0)
    metadata["config"]["accepted"] = False
    with pytest.raises(ValueError, match="did not pass"):
        load_rma_policy(
            checkpoint=output / "adaptation.flax",
            config=metadata["config"],
            deterministic=True,
        )


def test_critic_warmup_uses_reference_backup_even_when_q_prefers_learner(
    learner, monkeypatch
):
    from types import SimpleNamespace

    import learning.reinforcement.drlr2 as module

    state, _ = learner._step(learner.initialize())
    reference = jax.tree.map(lambda array: array[:2], state.replay)

    def sample(actor, params, observation, key):
        return jp.full((len(observation), 2), -0.3), jp.zeros(len(observation))

    def q(params, observation, action):
        value = jp.where(action[:, 0] < 0, 10.0, 1.0)
        return jp.stack((value, value), axis=-1)

    monkeypatch.setattr(module, "sample_action", sample)
    learner.critic = SimpleNamespace(apply=q)
    _, warmup = learner.gradient_update(state, reference)
    _, learning = learner.gradient_update(
        state.replace(vector_steps=jp.array(3)), reference
    )
    assert float(warmup["bellman_target_mean"]) == pytest.approx(-0.01 + 0.9995)
    assert float(learning["bellman_target_mean"]) == pytest.approx(-0.01 + 0.9995 * 10)
