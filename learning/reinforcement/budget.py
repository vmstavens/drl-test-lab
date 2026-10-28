"""Unambiguous training-budget accounting across skrl and Brax trainers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

BudgetUnit = Literal["transitions", "vector_steps"]


@dataclass(frozen=True)
class ResolvedTrainingBudget:
    """One requested budget represented in both common RL conventions."""

    requested_value: int
    requested_unit: BudgetUnit
    num_envs: int
    vector_steps: int
    transitions: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def resolve_training_budget(
    config: Mapping[str, Any],
    *,
    num_envs: int,
) -> ResolvedTrainingBudget:
    """Resolve ``transitions`` or skrl-style ``vector_steps`` to both units.

    ``timesteps`` remains a supported legacy alias for vector steps so existing
    experiment configurations retain their original sample count.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    raw = config.get("budget")
    if raw is None:
        if "timesteps" not in config:
            raise KeyError("RL config requires budget or legacy timesteps")
        value = int(config["timesteps"])
        unit: BudgetUnit = "vector_steps"
    else:
        value = int(raw["value"])
        unit = str(raw["unit"])
        if unit not in {"transitions", "vector_steps"}:
            raise ValueError("training budget unit must be transitions or vector_steps")
    if value < 1:
        raise ValueError("training budget value must be positive")
    if unit == "vector_steps":
        vector_steps = value
    else:
        vector_steps = math.ceil(value / num_envs)
    return ResolvedTrainingBudget(
        requested_value=value,
        requested_unit=unit,
        num_envs=num_envs,
        vector_steps=vector_steps,
        transitions=vector_steps * num_envs,
    )


def resolve_warmup_budget(
    config: Mapping[str, Any],
    *,
    num_envs: int,
) -> ResolvedTrainingBudget:
    """Resolve a warmup budget with legacy vector-step fallback."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    raw = config.get("warmup")
    if raw is None:
        value = int(
            config.get("warmup_vector_steps", config.get("warmup_timesteps", 0))
        )
        unit: BudgetUnit = "vector_steps"
    else:
        value = int(raw["value"])
        unit = str(raw["unit"])
        if unit not in {"transitions", "vector_steps"}:
            raise ValueError("warmup unit must be transitions or vector_steps")
    if value < 0:
        raise ValueError("warmup value must be non-negative")
    vector_steps = value if unit == "vector_steps" else math.ceil(value / num_envs)
    return ResolvedTrainingBudget(
        requested_value=value,
        requested_unit=unit,
        num_envs=num_envs,
        vector_steps=vector_steps,
        transitions=vector_steps * num_envs,
    )


__all__ = [
    "BudgetUnit",
    "ResolvedTrainingBudget",
    "resolve_training_budget",
    "resolve_warmup_budget",
]
