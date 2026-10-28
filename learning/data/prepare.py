"""Prepare immutable, hashed, episode-delimited transitions for DP training."""

import hashlib
import json
from pathlib import Path

import numpy as np

from learning.contracts import LearningContract
from learning.data.schema import DatasetManifest


def prepare_dataset(config, output_dir):
    source = Path(config["dataset"]["path"])
    contract = LearningContract.from_config(config)
    contract.require(json.loads((source / "collection.json").read_text())["contract"])
    destination = Path(config["transform"]["output_path"])
    if destination.exists():
        raise FileExistsError(
            f"Choose a fresh processed dataset directory: {destination}"
        )
    paths = sorted(source.glob("episode_*.npz"))
    if not paths:
        raise ValueError("Dataset contains no episodes")
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            episode = {name: data[name] for name in data.files}
        count = len(episode["actions"])
        if (
            count < 1
            or episode["observations"].shape
            != (count, len(contract.observation_layout))
            or episode["next_observations"].shape != episode["observations"].shape
            or episode["actions"].shape != (count, len(contract.action_low))
        ):
            raise ValueError(f"Trajectory dimensions differ from the contract: {path}")
        if (
            episode["terminated"].shape != (count,)
            or episode["rewards"].shape != (count,)
            or np.any(episode["terminated"][:-1])
        ):
            raise ValueError(f"Malformed episode boundaries: {path}")
        normalized = contract.normalize_action(episode["actions"])
        if np.any(np.abs(normalized) > 1.00001):
            raise ValueError(f"Executed actions exceed the declared bounds: {path}")
        if not bool(episode["terminated"][-1]) or episode["rewards"][-1] <= 0:
            continue
        if not all(np.isfinite(values).all() for values in episode.values()):
            raise ValueError(f"Nonfinite trajectory: {path}")
        episodes.append(episode)
    if not episodes:
        raise ValueError("Dataset contains no successful episodes")
    destination.mkdir(parents=True)
    names = {
        "observations": "states",
        "actions": "actions",
        "next_observations": "next_states",
        "rewards": "rewards",
        "terminated": "terminated",
    }
    for key, name in names.items():
        array = np.concatenate([episode[key] for episode in episodes])
        np.save(
            destination / f"{name}.npy",
            array.astype(bool if key == "terminated" else np.float32),
        )
    lengths = [len(episode["actions"]) for episode in episodes]
    np.save(destination / "episode_ends.npy", np.cumsum(lengths, dtype=np.int64))
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.glob("*.npy")
    }
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id=config["transform"]["dataset_id"],
        kind="processed",
        environment=contract.environment,
        observation_layout=contract.observation_layout,
        observation_dim=len(contract.observation_layout),
        action_dim=len(contract.action_low),
        action_semantics=contract.action_id,
        control_dt=contract.control_dt,
        sample_dt=contract.control_dt,
        episode_count=len(episodes),
        transition_count=sum(lengths),
        artifact_sha256=hashes,
        action_contract_id=contract.action_id,
        action_low=contract.action_low,
        action_high=contract.action_high,
        action_units=contract.action_units,
        observation_preprocessing=contract.to_dict(),
        source_dataset_ids=(config["dataset"]["id"],),
    )
    manifest.save(destination / "manifest.json")
    (output_dir / "prepared_dataset.json").write_text(
        json.dumps(
            {
                "path": str(destination),
                "episodes": len(episodes),
                "transitions": sum(lengths),
            },
            indent=2,
        )
    )
