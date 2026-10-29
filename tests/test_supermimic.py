"""Contract and numerical regressions for the independent research method."""

import copy
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from learning.policies.checkpoints import load_policy
from learning.reinforcement.task import Task
from learning.supermimic.objectives import lambda_returns, survival_return
from learning.supermimic.replay import SequenceReplay
from learning.supermimic.trainer import Trainer, validate_config

ROOT = Path(__file__).resolve().parents[1]


def small_config(method="supermimic"):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = OmegaConf.to_container(
            compose(config_name="train_supermimic"), resolve=True
        )
    config["task"]["env"]["episode_length"] = 6
    config["supermimic"].update(
        method=method,
        iterations=2,
        num_envs=3,
        holdout_envs=1,
        replay_steps=16,
        window_steps=2,
        horizon_steps=3,
        collect_steps=2,
        warmup_steps=3,
        batch_size=4,
        model_updates=1,
        critic_epochs=1,
        ppo_epochs=1,
        actor_widths=[8],
        model_widths=[8],
        critic_widths=[8],
        pixel_widths=[8],
    )
    config["supermimic"]["evaluation"].update(episodes=2)
    return config


def test_survival_reward_order_and_differentiable_termination():
    reward = jp.array([[2.0], [3.0], [100.0]])
    terminal = jp.array([[0.0], [1.0], [0.0]])
    np.testing.assert_allclose(
        survival_return(reward, terminal, jp.array([50.0]), 0.5), [3.5]
    )
    gradient = jax.grad(
        lambda term: survival_return(reward, term, jp.array([50.0]), 0.5).sum()
    )(terminal)
    assert np.isfinite(gradient).all()
    assert float(gradient[1, 0]) < 0
    targets = lambda_returns(reward, terminal, jp.zeros((4, 1)), 0.5, 1.0)
    np.testing.assert_allclose(targets[:, 0], [3.5, 3.0, 100.0])


def test_replay_wraparound_never_crosses_episode_or_holdout_lanes():
    replay = SequenceReplay(5, 3, 1)
    for t in range(12):
        replay.add(
            dict(
                state=np.array([[t, lane] for lane in range(3)]),
                done=np.full(3, t % 3 == 2),
            )
        )
    rng = np.random.default_rng(0)
    for holdout in (False, True):
        batch = replay.sample(rng, 100, 2, holdout=holdout)
        times = batch["state"][..., 0]
        lanes = batch["state"][..., 1]
        assert np.all(times >= 7)
        np.testing.assert_array_equal(np.diff(times, axis=1), 1)
        assert not batch["done"][:, :-1].any()
        assert np.all(lanes == 2) if holdout else np.all(lanes < 2)
    with pytest.raises(ValueError, match="episode-safe"):
        replay.sample(rng, 1, 4)


@pytest.fixture
def trainer():
    task = Task(small_config())
    learner = Trainer(small_config(), task)
    yield learner
    task.close()


def test_executed_replay_and_timeout_history(trainer):
    learner = trainer
    params, environment, previous, transition = learner._collect(
        learner.parameters,
        learner.environment,
        learner.previous,
        jp.zeros((learner.n, 0)),
        jp.full_like(learner.previous, 4.0),
        jp.array(True),
    )
    np.testing.assert_array_equal(transition["action"], 1.0)
    np.testing.assert_array_equal(transition["previous"], 0.0)
    np.testing.assert_array_equal(previous, 1.0)
    assert transition["state"].shape == (3, 4)
    for _ in range(5):
        params, environment, previous, transition = learner._collect(
            params,
            environment,
            previous,
            jp.zeros((learner.n, 0)),
            jp.full_like(previous, 4.0),
            jp.array(True),
        )
    np.testing.assert_array_equal(transition["done"], True)
    np.testing.assert_array_equal(transition["terminal"], 0.0)
    np.testing.assert_array_equal(previous, 0.0)
    np.testing.assert_array_equal(transition["next_previous"], 0.0)
    np.testing.assert_array_equal(environment.info["steps"], 0)
    assert len(learner._collect.cache) == 1


@pytest.mark.parametrize("method", ["supermimic", "ppo_sim", "ppo_wm"])
def test_compiled_method_updates_and_simulator_evaluation(method, tmp_path):
    config = small_config(method)
    task = Task(config)
    try:
        learner = Trainer(config, task)
        learner.collect(3, 0)
        before_actor = copy.deepcopy(learner.parameters.actor)
        before_model = copy.deepcopy(learner.parameters.model)
        metrics = learner.train_iteration(1)
        assert all(np.isfinite(float(value)) for value in metrics.values())
        assert any(
            not np.array_equal(x, y)
            for x, y in zip(
                jax.tree.leaves(before_actor),
                jax.tree.leaves(learner.parameters.actor),
                strict=True,
            )
        )
        model_changed = any(
            not np.array_equal(x, y)
            for x, y in zip(
                jax.tree.leaves(before_model),
                jax.tree.leaves(learner.parameters.model),
                strict=True,
            )
        )
        assert model_changed == (method != "ppo_sim")
        results = learner.evaluate()
        assert [r.seed for r in results] == [1000000, 1000001]
        assert all(np.isfinite(r.total_reward) for r in results)
        manifest = learner.save(tmp_path / "export")
        policy = load_policy(manifest, task.base)
        raw = task.base.reset(jax.random.PRNGKey(42))
        np.testing.assert_allclose(
            policy.act(raw.obs),
            learner.mean_action(
                learner.parameters.actor,
                raw.obs[None],
                jp.zeros((1, 2)),
                raw.info["privileged"][None],
                jp.zeros((1, 2)),
                None,
            )[0],
            atol=1e-6,
        )
        policy.observe_transition(np.array([0.2, -0.3]))
        np.testing.assert_allclose(policy.previous, [0.2, -0.3])
        policy.act(raw.obs)
        np.testing.assert_allclose(policy.previous, [0.2, -0.3])
        policy.reset(seed=0)
        np.testing.assert_array_equal(policy.previous, 0)
        incompatible = copy.deepcopy(task.base.learning_contract)
        incompatible["control_dt"] *= 2

        class WrongTask:
            observation_size = 4
            action_size = 2
            learning_contract = incompatible

        with pytest.raises(ValueError, match="semantic contract"):
            load_policy(manifest, WrongTask())
    finally:
        task.close()


def test_model_fit_reduces_multistep_error_and_does_not_update_actor(trainer):
    trainer.collect(5, 0)
    batch = trainer.sample(2)
    actor = copy.deepcopy(trainer.parameters.actor)
    before = float(trainer.model_loss(trainer.parameters.model, batch)[0])
    for _ in range(200):
        trainer.parameters, _ = trainer._model_update(trainer.parameters, batch)
    after = float(trainer.model_loss(trainer.parameters.model, batch)[0])
    assert after < before * 0.8
    for x, y in zip(
        jax.tree.leaves(actor), jax.tree.leaves(trainer.parameters.actor), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def test_analytic_actor_gradient_flows_through_frozen_model(trainer):
    trainer.collect(3, 0)
    starts = jax.tree.map(lambda x: x[:, 0], trainer.sample())
    original_model = copy.deepcopy(trainer.parameters.model)
    parameters, loss, _ = trainer._analytic_update(trainer.parameters, starts)
    assert np.isfinite(float(loss))
    assert any(
        not np.array_equal(x, y)
        for x, y in zip(
            jax.tree.leaves(parameters.actor),
            jax.tree.leaves(trainer.parameters.actor),
            strict=True,
        )
    )
    for x, y in zip(
        jax.tree.leaves(original_model), jax.tree.leaves(parameters.model), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def test_configuration_rejects_seed_overlap_and_incompatible_windows():
    config = small_config()
    config["supermimic"]["evaluation"]["seed_start"] = config["seed"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_config(config)
    config = small_config()
    config["supermimic"]["window_steps"] = 7
    with pytest.raises(ValueError, match="within one episode"):
        validate_config(config)


def test_hand_controller_projection_and_visual_privilege_boundary():
    from types import SimpleNamespace

    from envs.supermimic_hand import SuperMimicHand

    hand = SuperMimicHand.__new__(SuperMimicHand)
    hand._config = SimpleNamespace(action_scale=0.5)
    hand._lowers = jp.zeros(16)
    hand._uppers = jp.ones(16)
    state = jp.zeros(61).at[19].set(1.0).at[45:].set(0.9)
    realized = hand.model_action(state, jp.ones(16))
    np.testing.assert_allclose(realized, 0.2, atol=1e-6)
    projected = hand.model_transition(state, state, realized)
    np.testing.assert_allclose(projected[45:], 1.0)
    zero_quat = state.at[19:23].set(0)
    np.testing.assert_array_equal(hand.model_project(zero_quat)[19:23], [1, 0, 0, 0])
    assert np.isfinite(jax.jacrev(hand.model_project)(zero_quat)).all()
    learner = Trainer.__new__(Trainer)
    learner.task = SimpleNamespace(base=hand)
    learner.visual = True
    learner.goal_size = 4
    goal = jp.array([[1.0, 0.0, 0.0, 0.0]])
    context = jp.ones((1, 2))
    previous = jp.zeros((1, 16))
    original = learner.features(state[None], goal, context, previous)
    # All object pose/velocity and motor targets are privileged in the RGB arm.
    hidden_change = state.at[16:23].set(0.5).at[39:].set(0.25)
    changed = learner.features(hidden_change[None], goal, context * 5, previous)
    np.testing.assert_array_equal(original, changed)


def test_differentiable_image_head_supplies_object_gradients_to_visual_actor():
    from learning.supermimic.networks import Actor, PixelModel

    state = jp.ones((2, 6))
    context = jp.ones((2, 2))
    pixels = PixelModel((8, 8, 3), (8,))
    model = pixels.init(jax.random.PRNGKey(0), state, context)
    actor = Actor(2, (8,), visual=True)
    parameters = actor.init(
        jax.random.PRNGKey(1), jp.zeros((2, 4)), jp.zeros((2, 8, 8, 3))
    )

    def objective(s):
        image = pixels.apply(jax.lax.stop_gradient(model), s, context)
        return actor.apply(parameters, jp.zeros((2, 4)), image)[0].sum()

    gradient = jax.grad(objective)(state)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-6


def test_shared_frozen_model_changes_actor_but_preserves_source_model(
    trainer, tmp_path
):
    trainer.parameters = trainer.parameters.replace(
        model=jax.tree.map(lambda x: x * 0.75, trainer.parameters.model)
    )
    trainer.save(tmp_path / "shared")
    config = small_config("ppo_wm")
    config["supermimic"].update(
        model_manifest=str(tmp_path / "shared" / "world_model.json"), freeze_model=True
    )
    task = Task(config)
    try:
        learner = Trainer(config, task)
        original = copy.deepcopy(learner.parameters.model)
        for x, y in zip(
            jax.tree.leaves(original),
            jax.tree.leaves(trainer.parameters.model),
            strict=True,
        ):
            np.testing.assert_array_equal(x, y)
        learner.collect(3, 0)
        metrics = learner.train_iteration(1)
        assert np.isfinite(float(metrics["actor_loss"]))
        for x, y in zip(
            jax.tree.leaves(original),
            jax.tree.leaves(learner.parameters.model),
            strict=True,
        ):
            np.testing.assert_array_equal(x, y)
        config["supermimic"]["reward_scale"] = 2.0
        with pytest.raises(ValueError, match="semantic contract"):
            Trainer(config, task)
    finally:
        task.close()


def test_visual_policy_delay_queue_and_executed_history_reset():
    from learning.supermimic.networks import Actor
    from learning.supermimic.policy import SuperMimicPolicy

    actor = Actor(16, (8,), visual=True)
    params = actor.init(
        jax.random.PRNGKey(0), jp.zeros((1, 52)), jp.zeros((1, 8, 8, 3))
    )
    config = dict(
        observation="rgb",
        pixels={"shape": [8, 8, 3], "camera": "side"},
        contract={"action_low": [-0.5] * 16, "action_high": [0.5] * 16},
        observation_scale=[1.0] * 49,
        model_spec={"goal_layout": ["w", "x", "y", "z"]},
    )
    policy = SuperMimicPolicy(actor, params, config)
    for value in [0, 50, 100, 150]:
        action = policy.act(
            dict(
                pixels=np.full((8, 8, 3), value, dtype=np.uint8),
                proprioception=np.zeros(36),
                delay_steps=2,
            )
        )
        assert action.shape == (16,)
        assert np.isfinite(action).all()
    assert len(policy.pixel_queue) == 3
    np.testing.assert_array_equal(policy.pixel_queue[0], 50)
    np.testing.assert_array_equal(policy.previous, 0)
    policy.observe_transition(np.full(16, 0.25))
    np.testing.assert_allclose(policy.previous, 0.5)
    policy.reset(seed=42)
    assert policy.pixel_queue == []
    np.testing.assert_array_equal(policy.previous, 0)
    with pytest.raises(ValueError, match="Image dimensions"):
        policy.act(dict(pixels=np.zeros((16, 16, 3)), proprioception=np.zeros(36)))


def test_reset_preserves_warp_shared_buffers_without_a_gpu(trainer):
    from mujoco.mjx.warp.types import DATA_NON_VMAP, DataWarp

    from learning.supermimic.trainer import select_reset

    # Construct the pinned backend's real data structure without executing Warp.
    # Shared arrays deliberately have no environment dimension.
    def backend(value):
        return DataWarp(
            **{
                field.name: (
                    2
                    if field.type is int
                    else jp.full(
                        (5,) if field.name in DATA_NON_VMAP else (trainer.n, 1), value
                    )
                )
                for field in DataWarp.fields()
            }
        )

    old = trainer.environment.replace(
        data=trainer.environment.data.replace(_impl=backend(3.0))
    )
    fresh = old.replace(
        data=old.data.replace(qpos=old.data.qpos + 1, _impl=backend(9.0))
    )
    result = jax.jit(select_reset)(old, fresh, jp.array([True, False, True]))
    np.testing.assert_array_equal(result.data._impl.contact__dist, 3.0)
    np.testing.assert_array_equal(result.data._impl.cacc[:, 0], [9.0, 3.0, 9.0])
    np.testing.assert_allclose(result.data.qpos[0], fresh.data.qpos[0])
    np.testing.assert_allclose(result.data.qpos[1], old.data.qpos[1])
