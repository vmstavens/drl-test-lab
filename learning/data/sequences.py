"""Manifest-validated transition data and lazy diffusion-policy sequences."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learning.data.schema import DatasetManifest


@dataclass(frozen=True)
class TransitionDataset:
    path: Path
    manifest: DatasetManifest
    observations: np.ndarray
    actions: np.ndarray
    episode_ends: np.ndarray
    next_observations: np.ndarray | None = None
    rewards: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    active_starts: np.ndarray | None = None

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate((np.asarray([0], dtype=np.int64), self.episode_ends[:-1]))

    @property
    def expert_indices(self) -> np.ndarray:
        """Rows containing operator actions, excluding retained DP context."""
        active_starts = (
            self.episode_starts if self.active_starts is None else self.active_starts
        )
        return np.concatenate(
            [
                np.arange(start, end, dtype=np.int64)
                for start, end in zip(
                    active_starts,
                    self.episode_ends,
                    strict=True,
                )
            ]
        )


@dataclass(frozen=True)
class SequenceDataset:
    """A lazy list of valid sequence starts over immutable transition arrays."""

    transitions: TransitionDataset
    starts: np.ndarray
    obs_horizon: int
    pred_horizon: int

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("Batch indices must be one-dimensional.")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("Sequence batch index is out of range.")

        starts = self.starts[selected]
        observation_indices = starts[:, None] + np.arange(self.obs_horizon)
        action_starts = starts + self.obs_horizon - 1
        action_indices = action_starts[:, None] + np.arange(self.pred_horizon)
        return {
            "observations": np.asarray(
                self.transitions.observations[observation_indices],
                dtype=np.float32,
            ),
            "actions": np.asarray(
                self.transitions.actions[action_indices],
                dtype=np.float32,
            ),
        }


def load_processed_dataset(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> TransitionDataset:
    """Load arrays through their manifest and validate the complete contract."""
    path = Path(path).resolve()
    manifest = DatasetManifest.load(path / "manifest.json")
    if manifest.kind != "processed":
        raise ValueError(f"Expected a processed dataset, got {manifest.kind!r}.")

    if verify_hashes:
        for filename, expected_digest in manifest.artifact_sha256.items():
            actual_digest = _sha256(path / filename)
            if actual_digest != expected_digest:
                raise ValueError(f"Artifact hash mismatch for {filename}.")

    observations = np.load(path / "states.npy", mmap_mode="r")
    actions = np.load(path / "actions.npy", mmap_mode="r")
    episode_ends = np.load(path / "episode_ends.npy", mmap_mode="r")
    optional = {
        name: (
            np.load(path / filename, mmap_mode="r")
            if (path / filename).is_file()
            else None
        )
        for name, filename in (
            ("next_observations", "next_states.npy"),
            ("rewards", "rewards.npy"),
            ("terminated", "terminated.npy"),
            ("truncated", "truncated.npy"),
            ("active_starts", "active_starts.npy"),
        )
    }

    expected_observations = (manifest.transition_count, manifest.observation_dim)
    expected_actions = (manifest.transition_count, manifest.action_dim)
    if observations.shape != expected_observations:
        raise ValueError(
            f"states.npy has shape {observations.shape}; "
            f"manifest expects {expected_observations}."
        )
    if actions.shape != expected_actions:
        raise ValueError(
            f"actions.npy has shape {actions.shape}; "
            f"manifest expects {expected_actions}."
        )
    if observations.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError("Transition arrays must use float32.")
    if (
        episode_ends.shape != (manifest.episode_count,)
        or int(episode_ends[-1]) != manifest.transition_count
        or np.any(np.diff(episode_ends) <= 0)
    ):
        raise ValueError("episode_ends.npy does not match the manifest.")
    if optional["active_starts"] is not None:
        active_starts = optional["active_starts"]
        if (
            active_starts.shape != episode_ends.shape
            or np.any(active_starts < np.concatenate(([0], episode_ends[:-1])))
            or np.any(active_starts >= episode_ends)
        ):
            raise ValueError("active_starts.npy does not match episode boundaries.")
    for name in ("next_observations", "rewards", "terminated", "truncated"):
        value = optional[name]
        if value is not None and len(value) != manifest.transition_count:
            raise ValueError(f"{name} does not match the manifest transition count.")

    return TransitionDataset(
        path=path,
        manifest=manifest,
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
        **optional,
    )


def make_sequence_splits(
    transitions: TransitionDataset,
    *,
    obs_horizon: int,
    pred_horizon: int,
    validation_fraction: float,
    split_seed: int,
) -> tuple[SequenceDataset, SequenceDataset | None]:
    """Split whole episodes, then derive valid DP sequence starts."""
    if obs_horizon < 1 or pred_horizon < 1:
        raise ValueError("Sequence horizons must be positive.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")

    episode_count = transitions.manifest.episode_count
    episode_indices = np.arange(episode_count, dtype=np.int64)
    if validation_fraction == 0:
        train_starts = _sequence_starts(
            transitions,
            episode_indices,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
        )
        return (
            SequenceDataset(
                transitions,
                train_starts,
                obs_horizon,
                pred_horizon,
            ),
            None,
        )
    if episode_count < 2:
        raise ValueError("At least two episodes are required for validation.")

    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(episode_indices)
    validation_count = round(episode_count * validation_fraction)
    validation_count = min(max(1, validation_count), episode_count - 1)
    validation_episodes = np.sort(shuffled[:validation_count])
    train_episodes = np.sort(shuffled[validation_count:])

    train = SequenceDataset(
        transitions,
        _sequence_starts(
            transitions,
            train_episodes,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
        ),
        obs_horizon,
        pred_horizon,
    )
    validation = SequenceDataset(
        transitions,
        _sequence_starts(
            transitions,
            validation_episodes,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
        ),
        obs_horizon,
        pred_horizon,
    )
    if not len(train) or not len(validation):
        raise ValueError("The episode split produced an empty sequence dataset.")
    return train, validation


def _sequence_starts(
    transitions: TransitionDataset,
    episode_indices: np.ndarray,
    *,
    obs_horizon: int,
    pred_horizon: int,
) -> np.ndarray:
    window_length = obs_horizon + pred_horizon - 1
    starts: list[np.ndarray] = []
    episode_starts = transitions.episode_starts
    for episode_index in episode_indices:
        episode_start = int(episode_starts[episode_index])
        active_start = (
            episode_start
            if transitions.active_starts is None
            else int(transitions.active_starts[episode_index])
        )
        start = max(episode_start, active_start - obs_horizon + 1)
        end = int(transitions.episode_ends[episode_index])
        count = end - start - window_length + 1
        if count > 0:
            starts.append(np.arange(start, start + count, dtype=np.int64))
    if not starts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(starts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
