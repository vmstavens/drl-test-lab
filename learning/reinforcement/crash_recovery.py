"""Atomic recovery archives and TensorBoard restart markers for compiled DRLR."""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import flax.serialization
import jax
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event, SessionLog

_ARTIFACTS = ("best_evaluated_actor.flax", "best_training_state.flax")


def config_digest(config):
    # Runtime execution settings may change without changing the saved state contract.
    # They can still affect GPU numerical results and must be recorded with the run.
    values = {k: v for k, v in config.items() if k not in {"runtime", "output_dir"}}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _flatten_state(state):
    entries, structure = jax.tree_util.tree_flatten_with_path(state)
    leaves = tuple(value for _, value in entries)
    schema = [
        (
            jax.tree_util.keystr(key),
            list(np.shape(value)),
            str(value.dtype)
            if hasattr(value, "dtype")
            else str(np.asarray(value).dtype),
        )
        for key, value in entries
    ]
    return leaves, structure, schema


def save_recovery(path, state, metadata, config):
    """Commit state, progress and best-model artifacts as one atomic archive."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    leaves, _, schema = _flatten_state(state)
    metadata = dict(
        metadata, version=1, config_digest=config_digest(config), schema=schema
    )
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("metadata.json", json.dumps(metadata))
            archive.writestr("state.msgpack", flax.serialization.to_bytes(leaves))
            for name in _ARTIFACTS:
                artifact = path.parent / name
                if artifact.is_file():
                    archive.write(artifact, name)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_recovery(path, template, config, *, restore_artifacts=True):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata["version"] != 1 or metadata["config_digest"] != config_digest(
            config
        ):
            raise ValueError("Recovery checkpoint configuration differs from this run")
        leaves, structure, schema = _flatten_state(template)
        if metadata["schema"] != json.loads(json.dumps(schema)):
            raise ValueError(
                "Recovery state structure differs from initialized environment"
            )
        restored = flax.serialization.from_bytes(leaves, archive.read("state.msgpack"))
        state = jax.tree_util.tree_unflatten(structure, restored)
        if not restore_artifacts:
            return state, metadata
        # A crash may have left newer best files than the committed checkpoint.
        for name in _ARTIFACTS:
            if name in archive.namelist():
                destination = path.parent / name
                temporary = destination.with_suffix(".restore.tmp")
                with archive.open(name) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                os.replace(temporary, destination)
            else:
                (path.parent / name).unlink(missing_ok=True)
        return state, metadata


def mark_tensorboard_restart(writer, transitions, *, inclusive=False):
    """Hide orphaned events after the committed step, retaining its own events."""
    writer._event_file_writer.add_event(
        Event(
            wall_time=time.time(),
            step=int(transitions) + (0 if inclusive else 1),
            session_log=SessionLog(status=SessionLog.START),
        )
    )
    writer.flush()
