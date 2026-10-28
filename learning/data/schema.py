"""Versioned manifest contract for raw and processed trajectory datasets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    kind: str
    environment: str
    observation_layout: tuple[str, ...]
    observation_dim: int
    action_dim: int
    action_semantics: str
    control_dt: float
    episode_count: int
    sample_stride: int = 1
    sample_dt: float | None = None
    git_commit: str | None = None
    source_dataset_ids: tuple[str, ...] = ()
    transition_count: int = 0
    source_uri: str | None = None
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    action_contract_id: str | None = None
    action_low: tuple[float, ...] = ()
    action_high: tuple[float, ...] = ()
    action_units: tuple[str, ...] = ()
    observation_preprocessing: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported dataset schema_version {self.schema_version}; expected 1."
            )
        if self.kind not in {"raw", "processed"}:
            raise ValueError("Dataset kind must be 'raw' or 'processed'.")
        if not self.dataset_id or not self.environment:
            raise ValueError("dataset_id and environment must be non-empty.")
        if self.observation_dim != len(self.observation_layout):
            raise ValueError(
                "observation_dim does not match the observation_layout length."
            )
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive.")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive.")
        sample_dt = self.control_dt if self.sample_dt is None else float(self.sample_dt)
        if not math.isclose(
            sample_dt,
            self.control_dt * self.sample_stride,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "sample_dt must equal control_dt multiplied by sample_stride."
            )
        if self.episode_count < 0:
            raise ValueError("episode_count cannot be negative.")
        if self.transition_count < 0:
            raise ValueError("transition_count cannot be negative.")
        action_contract_values = (
            self.action_contract_id,
            self.action_low,
            self.action_high,
            self.action_units,
        )
        if any(action_contract_values):
            if not all(action_contract_values):
                raise ValueError(
                    "Action contract requires id, low, high, and units together."
                )
            if not (
                len(self.action_low)
                == len(self.action_high)
                == len(self.action_units)
                == self.action_dim
            ):
                raise ValueError(
                    "Action-contract bounds and units must match action_dim."
                )
            if any(
                high <= low
                for low, high in zip(
                    self.action_low,
                    self.action_high,
                    strict=True,
                )
            ):
                raise ValueError("Every action-contract upper bound must exceed lower.")
        for filename, digest in self.artifact_sha256.items():
            if not filename or len(digest) != 64:
                raise ValueError("artifact_sha256 must map file names to SHA-256 hex.")

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["observation_layout"] = tuple(payload["observation_layout"])
        payload["source_dataset_ids"] = tuple(payload.get("source_dataset_ids", ()))
        payload["artifact_sha256"] = dict(payload.get("artifact_sha256", {}))
        payload["action_low"] = tuple(payload.get("action_low", ()))
        payload["action_high"] = tuple(payload.get("action_high", ()))
        payload["action_units"] = tuple(payload.get("action_units", ()))
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def assert_compatible(
        self,
        *,
        environment: str,
        observation_dim: int,
        action_dim: int,
        control_dt: float,
        sample_stride: int | None = None,
        sample_dt: float | None = None,
    ) -> None:
        expected = {
            "environment": environment,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "control_dt": control_dt,
        }
        actual = {name: getattr(self, name) for name in expected}
        if sample_stride is not None:
            expected["sample_stride"] = sample_stride
            actual["sample_stride"] = self.sample_stride
        if sample_dt is not None:
            expected["sample_dt"] = sample_dt
            actual["sample_dt"] = (
                self.control_dt if self.sample_dt is None else self.sample_dt
            )
        mismatches = {}
        for name, value in expected.items():
            observed = actual[name]
            matches = (
                math.isclose(observed, value, rel_tol=1e-9, abs_tol=1e-12)
                if name in {"control_dt", "sample_dt"}
                else observed == value
            )
            if not matches:
                mismatches[name] = (observed, value)
        if mismatches:
            raise ValueError(f"Dataset/environment contract mismatch: {mismatches}")
