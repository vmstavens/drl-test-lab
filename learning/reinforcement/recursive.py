"""DP bootstrap, accepted-parent recursion and successful reference handoff."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
from skrl.utils.tensorboard import SummaryWriter

from learning.imitation.diffusion_policy import DPReference
from learning.reinforcement.budget import resolve_training_budget
from learning.reinforcement.crash_recovery import (
    config_digest,
    load_recovery,
    save_recovery,
)
from learning.reinforcement.drlr2 import Trainer, replay_from_paths
from learning.reinforcement.policy_rollouts import make_policy_rollout_runner
from learning.reinforcement.task import Task


def require_success(summary, *, count, rate=0.0):
    if summary["successes"] < count or summary["success_rate"] < rate:
        raise RuntimeError(
            f"Reference gate failed: {summary['successes']}/{summary['episodes']} "
            f"successes; requires {count} and {rate:.0%}"
        )


def collect_bootstrap(trainer):
    gate = trainer.rl["bootstrap"]

    def action(norm, params, obs, previous, valid, key):
        return trainer.teacher(
            trainer.features(obs, previous), obs[..., -trainer.o :], key
        )

    run = make_policy_rollout_runner(
        trainer.task,
        jax.jit(action),
        trainer.features,
        episodes=gate["episodes"],
        batch_size=trainer.rl["num_envs"],
        episode_length=trainer.config["task"]["env"]["episode_length"],
        action_size=trainer.a,
        execution="compiled",
        chunk_steps=trainer.rl["chunk_steps"],
        capture_successes=True,
    )
    summary, paths = run(
        SimpleNamespace(normalizer_state=(), policy_params=()),
        seed_start=gate["seed_start"],
    )
    require_success(
        summary, count=gate["minimum_successes"], rate=gate["minimum_success_rate"]
    )
    return summary, replay_from_paths(paths)


def evaluate_gate(trainer, state):
    """Progress never substitutes for the disjoint full learner-only gate."""
    gate = trainer.rl["evaluation_gate"]
    progress, _ = trainer.rollout(
        state.policy_params,
        episodes=gate["progress_episodes"],
        seed_start=gate["progress_seed_start"],
    )
    if progress["success_rate"] < gate["success_threshold"]:
        return {"accepted": False, "progress": progress}
    full, _ = trainer.rollout(
        state.policy_params, episodes=gate["episodes"], seed_start=gate["seed_start"]
    )
    return {
        "accepted": full["success_rate"] >= gate["success_threshold"],
        "progress": progress,
        "promotion": full,
    }


class ActorReference:
    def __init__(self, trainer, params):
        self.actor, self.params = trainer.actor, params
        import flax.serialization

        self.fingerprint = hashlib.sha256(
            flax.serialization.to_bytes(params)
            + np.asarray(trainer.center).tobytes()
            + np.asarray(trainer.scale).tobytes()
        ).hexdigest()
        self.center, self.scale = trainer.center, trainer.scale

    def __call__(self, raw, previous_raw, previous, factors, key):
        features = jp.concatenate(
            ((raw - self.center) / self.scale, previous, factors), axis=-1
        )
        mean, _ = self.actor.apply(self.params, features)
        return jp.tanh(mean)


def run_learning(
    trainer,
    state,
    successful_replay,
    directory,
    writer,
    *,
    accepted=False,
    next_evaluation=None,
    next_checkpoint=None,
):
    budget = resolve_training_budget(trainer.rl, num_envs=trainer.rl["num_envs"])
    gate = trainer.rl["evaluation_gate"]
    next_evaluation = (
        gate["minimum_transitions"] if next_evaluation is None else next_evaluation
    )
    next_checkpoint = (
        trainer.rl["checkpoint_interval_transitions"]
        if next_checkpoint is None
        else next_checkpoint
    )
    reference_hash = hashlib.sha256(
        (directory / "reference_replay.npz").read_bytes()
    ).hexdigest()
    while not accepted and int(state.vector_steps) < budget.vector_steps:
        steps = min(
            trainer.rl["chunk_steps"],
            budget.vector_steps - int(state.vector_steps),
        )
        state, metrics = trainer.train_chunk(state, successful_replay, steps)
        state, metrics = jax.block_until_ready((state, metrics))
        transitions = int(state.vector_steps) * trainer.rl["num_envs"]
        if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
            raise FloatingPointError("Nonfinite training metric")
        for name, value in metrics.items():
            writer.add_scalar(
                tag=f"Training assisted / {name}",
                value=float(value),
                timestep=transitions,
            )
        writer.add_scalar(
            tag="Entropy / Temperature",
            value=float(jp.exp(state.log_alpha)),
            timestep=transitions,
        )
        if transitions >= next_evaluation:
            result = evaluate_gate(trainer, state)
            (directory / "evaluation.json").write_text(json.dumps(result, indent=2))
            accepted = result["accepted"]
            writer.add_scalar(
                tag="Learner evaluation / Progress success",
                value=result["progress"]["success_rate"],
                timestep=transitions,
            )
            if "promotion" in result:
                writer.add_scalar(
                    tag="Learner evaluation / Promotion success",
                    value=result["promotion"]["success_rate"],
                    timestep=transitions,
                )
            next_evaluation += gate["interval_transitions"]
        if transitions >= next_checkpoint:
            save_recovery(
                directory / "recovery.zip",
                state,
                dict(
                    reference_hash=reference_hash,
                    reference_fingerprint=trainer.reference.fingerprint,
                    accepted=accepted,
                    next_evaluation=next_evaluation,
                    next_checkpoint=next_checkpoint
                    + trainer.rl["checkpoint_interval_transitions"],
                ),
                trainer.config,
            )
            trainer.save_actor(directory / "latest_actor.flax", state)
            next_checkpoint += trainer.rl["checkpoint_interval_transitions"]
        print(
            f"Training: {transitions}/{budget.transitions} transitions",
            flush=True,
        )
    save_recovery(
        directory / "recovery.zip",
        state,
        dict(
            accepted=accepted,
            reference_hash=reference_hash,
            reference_fingerprint=trainer.reference.fingerprint,
            next_evaluation=next_evaluation,
            next_checkpoint=next_checkpoint,
        ),
        trainer.config,
    )
    return state, accepted


def train_recursive(config, output_dir):
    if not config["il"]["checkpoint"]:
        raise ValueError("Supply il.checkpoint from the matching DP training workflow")
    levels = config["recursive"]["levels"]
    if (
        not levels
        or levels[0] != 0
        or any(b <= a for a, b in zip(levels, levels[1:], strict=False))
    ):
        raise ValueError("Recursive levels must start at zero and strictly increase")
    reference = None
    parent = None
    successful_replay = None
    records = []
    resume_path = config["recursive"].get("resume_from")
    start_index = 0
    parent_path = None
    if resume_path:
        resume_path = Path(resume_path).resolve()
        source = json.loads((resume_path / "stage_config.json").read_text())
        expected = copy.deepcopy(config)
        expected["recursive"]["resume_from"] = None
        expected["task"]["env"]["domain_level"] = source["task"]["env"]["domain_level"]
        if config_digest(expected) != config_digest(source):
            raise ValueError("Recovery changes the learning configuration")
        start_index = levels.index(source["task"]["env"]["domain_level"])
        with np.load(
            resume_path / "reference_replay.npz", allow_pickle=False
        ) as replay:
            successful_replay = {key: jp.asarray(replay[key]) for key in replay.files}
        parent_path = json.loads((resume_path / "reference.json").read_text())[
            "parent_stage"
        ]
        if parent_path:
            from learning.reinforcement.rma import load_phase1

            old_trainer, parent = load_phase1(parent_path)
            reference = ActorReference(old_trainer, parent)
            center, scale = old_trainer.center, old_trainer.scale
            old_trainer.task.close()
    for index, level in enumerate(levels):
        if index < start_index:
            continue
        directory = output_dir / f"stage_{index:03d}"
        directory.mkdir()
        stage = copy.deepcopy(config)
        stage["recursive"]["resume_from"] = None
        stage["task"]["env"]["domain_level"] = level
        (directory / "stage_config.json").write_text(json.dumps(stage, indent=2))
        task = Task(stage, domain_level=level)
        writer = SummaryWriter(str(directory / "training"))
        (directory / "reference.json").write_text(
            json.dumps({"parent_stage": parent_path}, indent=2)
        )
        try:
            if reference is None:
                reference = DPReference(
                    config["il"]["checkpoint"],
                    task.contract,
                    inference_steps=config["rl"]["teacher_diffusion_steps"],
                )
                center, scale = reference.center, reference.scale
            trainer = Trainer(stage, task, reference, center=center, scale=scale)
            state = trainer.initialize(parent=parent)
            if successful_replay is None:
                bootstrap, successful_replay = collect_bootstrap(trainer)
                (directory / "bootstrap.json").write_text(
                    json.dumps(bootstrap, indent=2)
                )
            np.savez(
                directory / "reference_replay.npz", **jax.device_get(successful_replay)
            )
            gate = trainer.rl["evaluation_gate"]
            accepted = False
            # An accepted parent may already solve this domain; validate it alone.
            if parent is not None and not (
                resume_path is not None and index == start_index
            ):
                initial, _ = trainer.rollout(
                    state.policy_params,
                    episodes=gate["episodes"],
                    seed_start=gate["seed_start"],
                )
                accepted = initial["success_rate"] >= gate["success_threshold"]
                (directory / "initial_evaluation.json").write_text(
                    json.dumps(initial, indent=2)
                )
            schedule = {}
            if resume_path is not None and index == start_index:
                state, saved = load_recovery(
                    resume_path / "recovery.zip", state, stage, restore_artifacts=False
                )
                if saved["reference_fingerprint"] != reference.fingerprint:
                    raise ValueError("Recovery reference checkpoint content changed")
                if (
                    saved["reference_hash"]
                    != hashlib.sha256(
                        (resume_path / "reference_replay.npz").read_bytes()
                    ).hexdigest()
                ):
                    raise ValueError("Recovery replay content changed")
                schedule = {
                    key: saved[key]
                    for key in ("accepted", "next_evaluation", "next_checkpoint")
                }
            else:
                schedule["accepted"] = accepted
            state, accepted = run_learning(
                trainer, state, successful_replay, directory, writer, **schedule
            )
            summary = dict(
                accepted=accepted,
                domain_level=level,
                transitions=int(state.vector_steps) * trainer.rl["num_envs"],
                evaluation_policy="learner_only",
            )
            if not accepted:
                (directory / "summary.json").write_text(json.dumps(summary, indent=2))
                raise RuntimeError(
                    f"Learner failed acceptance at domain level {level}; "
                    "recursion stopped"
                )
            trainer.save_actor(directory / "accepted_actor.flax", state)
            # Replay is collected on training seeds only, never evaluation seeds.
            handoff = trainer.rl["handoff"]
            handoff_summary, paths = trainer.rollout(
                state.policy_params,
                episodes=handoff["episodes"],
                seed_start=handoff["seed_start"],
                capture=True,
            )
            require_success(handoff_summary, count=handoff["minimum_successes"])
            added = replay_from_paths(paths)
            successful_replay = jax.tree.map(
                lambda old, new, capacity=trainer.rl["memory_size"]: jp.concatenate(
                    (old, new)
                )[-capacity:],
                successful_replay,
                added,
            )
            np.savez(
                directory / "handoff_replay.npz", **jax.device_get(successful_replay)
            )
            summary["handoff"] = handoff_summary
            (directory / "summary.json").write_text(json.dumps(summary, indent=2))
            records.append(summary)
            (output_dir / "summary.json").write_text(json.dumps(records, indent=2))
            parent = state.policy_params
            reference = ActorReference(trainer, parent)
            parent_path = str(directory.resolve())
        finally:
            writer.close()
            task.close()
