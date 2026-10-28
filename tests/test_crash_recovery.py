"""Recovery must preserve trajectory state and hide events from lost progress."""

import dataclasses

import jax
import numpy as np
import pytest
from skrl.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from learning.reinforcement.crash_recovery import (
    load_recovery,
    mark_tensorboard_restart,
    save_recovery,
)


# Simulates MJX data: registered as a JAX pytree, not a Flax serializable class.
@dataclasses.dataclass
class SimulatorState:
    position: object
    episode_steps: object


jax.tree_util.register_dataclass(SimulatorState)


def state(step):
    return {
        "env": SimulatorState(np.array([step, 2.0]), np.array(step)),
        "replay": np.arange(12, dtype=np.float32).reshape(3, 4),
        "rng": np.array([12, 34], dtype=np.uint32),
        "step": np.array(step),
    }


def test_roundtrip_restores_simulator_replay_progress_and_best(tmp_path):
    path = tmp_path / "recovery.zip"
    best = tmp_path / "best_evaluated_actor.flax"
    best.write_bytes(b"checkpoint best")
    config = {"rl": {"budget": 100}, "runtime": {"console_metrics": True}}
    save_recovery(path, state(30), {"next_evaluation_transition": 40}, config)
    best.write_bytes(b"uncommitted best")
    restored, metadata = load_recovery(path, state(0), dict(config, runtime={}))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored),
        jax.tree_util.tree_leaves(state(30)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert metadata["next_evaluation_transition"] == 40
    assert best.read_bytes() == b"checkpoint best"
    with pytest.raises(ValueError, match="configuration differs"):
        load_recovery(path, state(0), {"rl": {"budget": 200}})


def test_failed_commit_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "recovery.zip"
    save_recovery(path, state(10), {}, {})
    import learning.reinforcement.crash_recovery as recovery

    def fail(*args):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(recovery.os, "replace", fail)
    with pytest.raises(OSError):
        save_recovery(path, state(20), {}, {})
    restored, _ = load_recovery(path, state(0), {})
    assert restored["step"] == 10
    assert not path.with_suffix(".tmp").exists()


def test_tensorboard_resume_removes_only_orphaned_steps(tmp_path):
    writer = SummaryWriter(str(tmp_path))
    for step in (10, 20, 30):
        writer.add_scalar(tag="reward", value=step, timestep=step)
    writer.close()
    writer = SummaryWriter(str(tmp_path))
    mark_tensorboard_restart(writer, 20)
    writer.add_scalar(tag="reward", value=99, timestep=30)
    writer.close()
    events = EventAccumulator(str(tmp_path)).Reload().Scalars("reward")
    assert [(event.step, event.value) for event in events] == [
        (10, 10),
        (20, 20),
        (30, 99),
    ]
