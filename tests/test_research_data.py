"""Episode boundaries, held-out splits and immutable dataset hashes."""

import json

import numpy as np
import pytest

from learning.contracts import LearningContract
from learning.data.prepare import prepare_dataset
from learning.data.sequences import load_processed_dataset, make_sequence_splits
from tests.test_learning_infrastructure import compose_workflow


def test_preparation_preserves_episodes_and_splits_then_rejects_corruption(tmp_path):
    config = compose_workflow("prepare")
    raw, processed, output = (
        tmp_path / "raw",
        tmp_path / "processed",
        tmp_path / "output",
    )
    raw.mkdir()
    output.mkdir()
    config["dataset"]["path"] = str(raw)
    config["transform"]["output_path"] = str(processed)
    contract = LearningContract.from_config(config)
    (raw / "collection.json").write_text(json.dumps({"contract": contract.to_dict()}))
    for episode in range(3):
        observations = np.full((32, 4), episode, np.float32)
        np.savez(
            raw / f"episode_{episode:03d}.npz",
            observations=observations,
            actions=np.full((32, 2), 0.1, np.float32),
            next_observations=observations,
            rewards=np.r_[np.full(31, -0.01), 99.99].astype(np.float32),
            terminated=np.arange(32) == 31,
        )
    prepare_dataset(config, output)
    dataset = load_processed_dataset(processed)
    train, validation = make_sequence_splits(
        dataset,
        obs_horizon=2,
        pred_horizon=16,
        validation_fraction=1 / 3,
        split_seed=42,
    )
    assert dataset.manifest.episode_count == 3
    assert set(train.starts // 32).isdisjoint(set(validation.starts // 32))
    for sequence in (train, validation):
        assert np.all(sequence.starts // 32 == (sequence.starts + 16) // 32)
    actions = np.load(processed / "actions.npy")
    actions[0, 0] = 0.2
    np.save(processed / "actions.npy", actions)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_processed_dataset(processed)
