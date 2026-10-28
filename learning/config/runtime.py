"""Conversion and provenance boundary between Hydra and workflow code."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from learning.notifications import (
    WorkflowNotification,
    send_workflow_notification,
)
from learning.publication.provenance import finish_run, start_run
from learning.tensorboard import tensorboard_session


def resolved_mapping(config: DictConfig) -> dict[str, Any]:
    """Resolve interpolations and reject missing mandatory values."""
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing mandatory configuration values: {sorted(missing)}")
    OmegaConf.resolve(config)
    result = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise TypeError("The composed workflow configuration must be a mapping.")
    if result.get("contract"):
        from learning.contracts import LearningContract

        contract = LearningContract.from_config(result)
        expected = result.get("dataset", {}).get("expected", {})
        if expected and (
            expected["observation_dim"] != len(contract.observation_layout)
            or expected["action_dim"] != len(contract.action_low)
            or expected["control_dt"] != contract.control_dt
        ):
            raise ValueError(
                "Dataset dimensions/timing differ from the learning contract"
            )
    return result


def execute_workflow(
    config: DictConfig,
    implementation: Callable[[Mapping[str, Any], Path], None],
) -> None:
    """Persist the resolved config, then dry-run or execute one stage."""
    values = resolved_mapping(config)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "provenance.json").exists():
        raise FileExistsError(
            f"Choose a new run directory; {output_dir} already has a run."
        )
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )

    runtime = values.get("runtime", {})
    print(f"Run directory: {output_dir}", flush=True)
    training = str(values.get("stage", "")).startswith("train_")
    tensorboard = runtime.get("tensorboard", {}) if training else {"mode": "off"}
    log_dir = Path(tensorboard.get("log_dir") or output_dir)
    dry_run = runtime.get("dry_run", False)
    record = None if dry_run else start_run(values, output_dir)
    try:
        with tensorboard_session(tensorboard, log_dir, dry_run=dry_run):
            _run_or_preview(values, output_dir, implementation)
    except BaseException as error:
        if record is not None:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            finish_run(output_dir, record, status, type(error).__name__)
        raise
    else:
        if record is not None:
            finish_run(output_dir, record, "succeeded")


def _run_or_preview(
    values: Mapping[str, Any],
    output_dir: Path,
    implementation: Callable[[Mapping[str, Any], Path], None],
) -> None:
    runtime = values.get("runtime", {})
    if runtime.get("dry_run", False):
        stage = values.get("stage", "workflow")
        print(f"DRY RUN: composed and validated {stage!r}.")
        return
    _execute_implementation(values, output_dir, implementation)


def _execute_implementation(
    values: Mapping[str, Any],
    output_dir: Path,
    implementation: Callable[[Mapping[str, Any], Path], None],
) -> None:
    """Execute one stage and report its terminal state on a best-effort basis."""
    started = time.monotonic()
    status = "succeeded"
    error_type = None
    try:
        if values.get("contract") or values.get("stage") in {
            "evaluate",
            "view_policy",
            "render_rollout",
            "view_env",
        }:
            import jax

            backend = "cpu" if values["runtime"]["device"] == "cpu" else "gpu"
            with jax.default_device(jax.devices(backend)[0]):
                implementation(values, output_dir)
        else:
            implementation(values, output_dir)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        error_type = type(error).__name__
        raise
    finally:
        runtime = values.get("runtime", {})
        notifications = (
            runtime.get("notifications") if isinstance(runtime, Mapping) else None
        )
        task = values.get("task", {})
        task_name = (
            str(task.get("name", "unknown")) if isinstance(task, Mapping) else "unknown"
        )
        send_workflow_notification(
            notifications,
            WorkflowNotification(
                status=status,
                stage=str(values.get("stage", "workflow")),
                task=task_name,
                duration_seconds=time.monotonic() - started,
                output_dir=output_dir,
                hostname=socket.gethostname(),
                error_type=error_type,
            ),
        )
