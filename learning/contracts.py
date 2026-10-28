"""One semantic contract for datasets, policies, controllers and checkpoints."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LearningContract:
    environment: str
    observation_layout: tuple[str, ...]
    action_id: str
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    action_units: tuple[str, ...]
    control_dt: float
    sim_dt: float
    coordinate_frame: str
    privileged_layout: tuple[str, ...] = ()
    preprocessing: str = "identity_v1"
    previous_action_source: str = "executed"

    @classmethod
    def from_config(cls, config):
        c = config["contract"]
        result = cls(
            environment=config["task"]["env"]["registry_name"],
            observation_layout=tuple(c["observation_layout"]),
            action_id=c["action_id"],
            action_low=tuple(c["action_low"]),
            action_high=tuple(c["action_high"]),
            action_units=tuple(c["action_units"]),
            control_dt=float(config["task"]["env"]["ctrl_dt"]),
            sim_dt=float(config["task"]["env"]["sim_dt"]),
            coordinate_frame=c["coordinate_frame"],
            privileged_layout=tuple(c.get("privileged_layout", ())),
            preprocessing=c["preprocessing"],
        )
        result.validate()
        return result

    def validate(self):
        low, high = np.asarray(self.action_low), np.asarray(self.action_high)
        if (
            low.ndim != 1
            or not len(low)
            or low.shape != high.shape
            or len(low) != len(self.action_units)
        ):
            raise ValueError(
                "Action bounds and units must have equal positive dimensions"
            )
        if (
            not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(high <= low)
        ):
            raise ValueError("Action bounds must be finite and strictly ordered")
        ratio = self.control_dt / self.sim_dt if self.sim_dt > 0 else 0
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("Control interval must contain whole simulation steps")
        if self.previous_action_source != "executed" or not self.observation_layout:
            raise ValueError(
                "Executed-action history and observation layout are required"
            )

    def to_dict(self):
        import json

        return json.loads(json.dumps(asdict(self)))

    def require(self, recorded):
        if self.to_dict() != recorded:
            raise ValueError(
                "Learning contract differs: frame, timing, units or preprocessing"
            )

    def normalize_action(self, physical):
        return (
            2
            * (physical - np.asarray(self.action_low))
            / (np.asarray(self.action_high) - self.action_low)
            - 1
        )


def append_history(observation, executed):
    import jax.numpy as jp

    return jp.concatenate((observation, executed), axis=-1)
