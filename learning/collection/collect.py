"""Collect successful scripted demonstrations with the shared rolling runner."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np

from learning.reinforcement.policy_rollouts import make_policy_rollout_runner
from learning.reinforcement.task import Task


def collect_demonstrations(config, output_dir):
    if config["collector"]["controller"] != "scripted":
        raise ValueError("The worked example supports the scripted controller")
    task = Task(config)
    destination = Path(config["dataset"]["path"])
    if destination.exists():
        raise FileExistsError(f"Choose a fresh dataset directory: {destination}")
    try:
        low, high = (
            jp.asarray(task.contract.action_low),
            jp.asarray(task.contract.action_high),
        )

        def expert(norm, params, obs, previous, valid, key):
            physical = task.base.expert(obs)
            return jp.clip(2 * (physical - low) / (high - low) - 1, -1, 1)

        run = make_policy_rollout_runner(
            task,
            jax.jit(expert),
            lambda obs, previous: obs[:, : task.observation_size],
            episodes=int(config["collector"]["episodes"]),
            batch_size=int(config["collector"]["batch_size"]),
            episode_length=int(config["collector"]["max_steps"]),
            action_size=task.action_size,
            execution="compiled",
            chunk_steps=64,
            capture_successes=True,
        )
        summary, paths = run(
            SimpleNamespace(normalizer_state=(), policy_params=()),
            seed_start=int(config["seed"]),
        )
        if not paths:
            raise RuntimeError(
                "Scripted controller produced no successful demonstrations"
            )
        destination.mkdir(parents=True)
        for index, (length, obs, actions, following, rewards) in enumerate(paths):
            physical = low + (actions + 1) * 0.5 * (high - low)
            np.savez(
                destination / f"episode_{index:06d}.npz",
                observations=obs,
                actions=np.asarray(physical),
                next_observations=following,
                rewards=rewards,
                terminated=np.arange(length) == length - 1,
            )
        metadata = {
            "contract": task.contract.to_dict(),
            "summary": summary,
            "seed_start": int(config["seed"]),
        }
        (destination / "collection.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / "collection_summary.json").write_text(
            json.dumps(metadata, indent=2)
        )
    finally:
        task.close()
