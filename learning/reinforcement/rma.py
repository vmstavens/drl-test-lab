"""Separate DAgger history adaptation with a frozen actor and privileged encoder."""

import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jp
import numpy as np
import optax
from mjxsim.rma import RmaSpec
from mjxsim.rma.jax import AdaptationEncoder
from skrl.utils.tensorboard import SummaryWriter

from learning.reinforcement.drlr2 import Trainer
from learning.reinforcement.task import Task


def load_phase1(directory):
    directory = Path(directory)
    if not json.loads((directory / "summary.json").read_text())["accepted"]:
        raise ValueError("RMA requires an accepted phase-one learner")
    path = directory / "accepted_actor.flax"
    metadata = json.loads(path.with_suffix(".json").read_text())
    config = metadata["config"]
    task = Task(config, domain_level=config["task"]["env"]["domain_level"])
    task.contract.require(metadata["contract"])
    trainer = Trainer(
        config, task, None, center=metadata["center"], scale=metadata["scale"]
    )
    template = trainer.actor.init(jax.random.PRNGKey(0), jp.zeros((1, trainer.width)))
    return trainer, flax.serialization.from_bytes(template, path.read_bytes())


def adaptation_module(trainer, settings):
    spec = RmaSpec(
        trainer.o,
        trainer.a,
        trainer.f,
        trainer.rl["latent_dim"],
        settings["history_length"],
    )
    return AdaptationEncoder(
        spec, step_dim=settings["step_dim"], temporal_dim=settings["temporal_dim"]
    )


def student_action(trainer, frozen, adaptation, params, obs, previous, history):
    """The student's actor receives no privileged factors."""
    latent = adaptation.apply(params, history)
    features = trainer.features(obs, previous).at[..., trainer.o + trainer.a :].set(0.0)
    mean, _ = trainer.actor.apply(frozen, features, latent=latent)
    return jp.tanh(mean)


def advance_history(history, feature, done, sample):
    shifted = jp.concatenate((history[:, 1:], feature[:, None]), axis=1)
    return jp.where(
        done[:, None, None],
        0.0,
        jp.where(jp.asarray(sample)[..., None, None], shifted, history),
    )


def evaluate_student(trainer, frozen, adaptation, params, settings):
    """Held-out student-only evaluation; no privileged-teacher inference."""
    horizon = trainer.config["task"]["env"]["episode_length"]

    @jax.jit
    def run(keys):
        state = trainer.task.reset(keys)
        n = len(keys)
        previous = jp.zeros((n, trainer.a))
        history = jp.zeros((n, settings["history_length"], trainer.o + trainer.a))

        def body(carry, index):
            state, previous, history, active, successes = carry
            action = student_action(
                trainer, frozen, adaptation, params, state.obs, previous, history
            )
            following = trainer.task.step(state, action)
            executed, done = (
                trainer.task.executed(following),
                following.done.astype(bool),
            )
            successes |= active & done & (following.reward > 0)
            feature = jp.concatenate(
                (
                    (state.obs[..., : trainer.o] - trainer.center) / trainer.scale,
                    executed,
                ),
                axis=-1,
            )
            history = advance_history(
                history,
                feature,
                done,
                state.info["steps"] % settings["sample_stride"] == 0,
            )
            return (
                following,
                jp.where(done[:, None], 0.0, executed),
                history,
                active & ~done,
                successes,
            ), None

        result, _ = jax.lax.scan(
            body,
            (state, previous, history, jp.ones(n, bool), jp.zeros(n, bool)),
            jp.arange(horizon),
        )
        return result[-1]

    results = []
    batch_size = min(
        settings["evaluation_episodes"], trainer.rl["evaluation_gate"]["batch_size"]
    )
    for offset in range(0, settings["evaluation_episodes"], batch_size):
        size = min(batch_size, settings["evaluation_episodes"] - offset)
        keys = jax.vmap(jax.random.PRNGKey)(
            jp.arange(size) + settings["seed_start"] + offset
        )
        results.extend(np.asarray(run(keys)).tolist())
    return dict(
        episodes=len(results),
        successes=sum(results),
        success_rate=float(np.mean(results)),
        evaluation_policy="learner_only",
        conditioning="observable_history",
    )


def train_phase2(config, output_dir):
    if not config["phase1_job"]:
        raise ValueError("Supply phase1_job pointing to an accepted recursive stage")
    trainer, frozen = load_phase1(config["phase1_job"])
    settings = config["rma"]
    if not settings["probabilities"] or settings["probabilities"][-1] != 1.0:
        raise ValueError("DAgger must end with student-only collection")
    if settings["sample_stride"] < 1 or settings["history_length"] < 1:
        raise ValueError("History length and stride must be positive")
    adaptation = adaptation_module(trainer, settings)
    key, init_key, reset_key = jax.random.split(jax.random.PRNGKey(config["seed"]), 3)
    history = jp.zeros(
        (settings["num_envs"], settings["history_length"], trainer.o + trainer.a)
    )
    params = adaptation.init(init_key, history)
    optimizer = optax.adam(settings["learning_rate"])
    optimizer_state = optimizer.init(params)
    state = trainer.task.reset(jax.random.split(reset_key, settings["num_envs"]))
    previous = jp.zeros((settings["num_envs"], trainer.a))

    @jax.jit
    def collect(state, previous, history, key, params, probability):
        def body(carry, index):
            state, previous, history, key = carry
            key, select_key, reset_key = jax.random.split(key, 3)
            features = trainer.features(state.obs, previous)
            target = trainer.actor.apply(
                frozen,
                features[..., trainer.o + trainer.a :],
                method=trainer.actor.encode,
            )
            teacher_mean, _ = trainer.actor.apply(frozen, features)
            learner = student_action(
                trainer, frozen, adaptation, params, state.obs, previous, history
            )
            use_student = jax.random.bernoulli(
                select_key, probability, (settings["num_envs"], 1)
            )
            following = trainer.task.step(
                state, jp.where(use_student, learner, jp.tanh(teacher_mean))
            )
            executed, done = (
                trainer.task.executed(following),
                following.done.astype(bool),
            )
            feature = jp.concatenate((features[..., : trainer.o], executed), axis=-1)
            next_history = advance_history(
                history,
                feature,
                done,
                state.info["steps"] % settings["sample_stride"] == 0,
            )
            fresh = trainer.task.reset(
                jax.random.split(reset_key, settings["num_envs"])
            )

            def reset(a, b):
                if hasattr(a, "ndim") and a.ndim and a.shape[0] == len(done):
                    return jp.where(
                        done.reshape((len(done),) + (1,) * (a.ndim - 1)), b, a
                    )
                return a

            return (
                jax.tree.map(reset, following, fresh),
                jp.where(done[:, None], 0.0, executed),
                next_history,
                key,
            ), (history, target)

        return jax.lax.scan(
            body,
            (state, previous, history, key),
            jp.arange(settings["collection_steps"]),
        )

    @jax.jit
    def update(params, optimizer_state, histories, targets):
        def loss(params):
            return jp.mean((adaptation.apply(params, histories) - targets) ** 2)

        value, gradient = jax.value_and_grad(loss)(params)
        changes, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        return optax.apply_updates(params, changes), optimizer_state, value

    histories, targets, records = None, None, []
    writer = SummaryWriter(str(output_dir / "training"))
    try:
        for round_index, probability in enumerate(settings["probabilities"]):
            (state, previous, history, key), examples = collect(
                state, previous, history, key, params, jp.asarray(probability)
            )
            new_histories, new_targets = jax.device_get(examples)
            new_histories = new_histories[:: settings["sample_stride"]].reshape(
                (-1, *history.shape[1:])
            )
            new_targets = new_targets[:: settings["sample_stride"]].reshape(
                (-1, trainer.rl["latent_dim"])
            )
            histories = (
                new_histories
                if histories is None
                else np.concatenate((histories, new_histories))
            )
            targets = (
                new_targets
                if targets is None
                else np.concatenate((targets, new_targets))
            )
            histories, targets = (
                histories[-settings["replay_capacity"] :],
                targets[-settings["replay_capacity"] :],
            )
            losses = []
            rng = np.random.default_rng(config["seed"] + round_index)
            for _ in range(settings["updates_per_round"]):
                indices = rng.integers(len(histories), size=settings["batch_size"])
                params, optimizer_state, loss = update(
                    params, optimizer_state, histories[indices], targets[indices]
                )
                losses.append(float(loss))
            mse = float(np.mean(losses))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite RMA distillation loss")
            records.append(
                dict(
                    round=round_index + 1,
                    student_probability=probability,
                    adaptation_mse=mse,
                    replay_size=len(histories),
                )
            )
            writer.add_scalar(
                tag="RMA / Adaptation MSE", value=mse, timestep=round_index + 1
            )
            print(records[-1], flush=True)
        evaluation = evaluate_student(trainer, frozen, adaptation, params, settings)
        accepted = (
            evaluation["success_rate"]
            >= trainer.rl["evaluation_gate"]["success_threshold"]
        )
        (output_dir / "adaptation.flax").write_bytes(
            flax.serialization.to_bytes(params)
        )
        (output_dir / "actor.flax").write_bytes(flax.serialization.to_bytes(frozen))
        metadata = dict(
            schema_version=1,
            policy_type="rma_history",
            loader="learning.policies.research:load_rma_policy",
            checkpoint="adaptation.flax",
            action_semantics=trainer.task.contract.action_id,
            observation_dim=trainer.o,
            action_dim=trainer.a,
            config=dict(
                contract=trainer.task.contract.to_dict(),
                stage=trainer.config,
                rma=settings,
                center=np.asarray(trainer.center).tolist(),
                scale=np.asarray(trainer.scale).tolist(),
                accepted=accepted,
            ),
        )
        (output_dir / "policy.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / "summary.json").write_text(
            json.dumps(
                dict(accepted=accepted, student_evaluation=evaluation, rounds=records),
                indent=2,
            )
        )
        if not accepted:
            raise RuntimeError(
                "History-adapted learner failed its held-out acceptance gate"
            )
    finally:
        writer.close()
        trainer.task.close()
