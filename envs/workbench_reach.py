"""Small MuJoCo reaching task for exercising the complete research pipeline."""

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.registry import register_environment

XML = """<mujoco><option timestep="0.0005" gravity="0 0 0"/>
<worldbody><light pos="0 0 2"/><camera name="top" pos="0 0 1.5"/>
<geom type="plane" size="1 1 .01" rgba=".2 .2 .2 1"/>
<site name="target" pos="0 0 .04" size=".02" rgba="0 1 0 .5"/>
<body pos="0 0 .04"><joint name="x" type="slide" axis="1 0 0" damping="1"/>
<joint name="y" type="slide" axis="0 1 0" damping="1"/>
<geom type="sphere" size=".025" mass=".05" contype="0" conaffinity="0"/></body>
</worldbody><actuator><motor joint="x"/><motor joint="y"/></actuator></mujoco>"""


def default_config():
    return config_dict.create(
        ctrl_dt=0.002,
        sim_dt=0.0005,
        episode_length=1000,
        action_repeat=1,
        impl="jax",
        domain_level=0.0,
    )


class WorkbenchReach(mjx_env.MjxEnv):
    """Reach the origin with uncertain actuator gain and viscous load.

    Observable state: x/y position and velocity. Privileged factors: gain and
    added damping, normalized to the fixed full-domain bounds. Commands are
    forces in N; four physics substeps execute each 2 ms control command.
    """

    observation_layout = ("x_m", "y_m", "vx_m_s", "vy_m_s")

    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        if self._config.impl != "jax":
            raise ValueError("WorkbenchReach uses the portable MJX JAX backend")
        self._mj_model = mujoco.MjModel.from_xml_string(XML)
        self._mj_model.opt.timestep = self._sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._substeps = round(self._ctrl_dt / self._sim_dt)
        if (
            self._substeps < 1
            or abs(self._substeps * self._sim_dt - self._ctrl_dt) > 1e-10
        ):
            raise ValueError("Control interval must contain whole simulation steps")
        self.domain_level = float(self._config.domain_level)
        if not 0 <= self.domain_level <= 1:
            raise ValueError("domain_level must be in [0, 1]")

    @property
    def learning_contract(self):
        return dict(
            environment="WorkbenchReach",
            observation_layout=list(self.observation_layout),
            action_id="workbench_force_xy_v1",
            action_low=[-1.0, -1.0],
            action_high=[1.0, 1.0],
            action_units=["N", "N"],
            control_dt=self._ctrl_dt,
            sim_dt=self._sim_dt,
            coordinate_frame="world_xy",
            privileged_layout=["actuator_gain_normalized", "viscous_load_normalized"],
            preprocessing="identity_v1",
            previous_action_source="executed",
        )

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def xml_path(self):
        return "generated://workbench_reach"

    @property
    def action_size(self):
        return 2

    def reset(self, rng):
        position_key, factor_key = jax.random.split(rng)
        position = jax.random.uniform(position_key, (2,), minval=-0.2, maxval=0.2)
        factors = (
            jax.random.uniform(factor_key, (2,), minval=-1.0, maxval=1.0)
            * self.domain_level
        )
        data = mjx.make_data(self.mjx_model).replace(qpos=position)
        data = mjx.forward(self.mjx_model, data)
        info = {
            "privileged": factors,
            "steps": jp.array(0),
            "executed_action": jp.zeros(2),
        }
        return mjx_env.State(
            data,
            self._obs(data),
            jp.array(0.0),
            jp.array(0.0),
            {"is_success": jp.array(0.0)},
            info,
        )

    @staticmethod
    def _obs(data):
        return jp.concatenate((data.qpos, data.qvel))

    def step(self, state, action):
        command = jp.clip(action, -1.0, 1.0)
        gain = 1.0 + 0.4 * state.info["privileged"][0]
        damping = 0.5 * (state.info["privileged"][1] + 1.0)

        def physics(data, _):
            data = data.replace(ctrl=command * gain, qfrc_applied=-damping * data.qvel)
            return mjx.step(self.mjx_model, data), None

        data, _ = jax.lax.scan(physics, state.data, None, length=self._substeps)
        steps = state.info["steps"] + 1
        success = (jp.linalg.norm(data.qpos) < 0.015) & (
            jp.linalg.norm(data.qvel) < 0.03
        )
        finite = jp.all(jp.isfinite(self._obs(data)))
        done = success | (steps >= self._config.episode_length) | ~finite
        return state.replace(
            data=data,
            obs=self._obs(data),
            reward=jp.where(success, 99.99, -0.01),
            done=done.astype(jp.float32),
            metrics={"is_success": success.astype(jp.float32)},
            info={**state.info, "steps": steps, "executed_action": command},
        )

    @staticmethod
    def expert(observation):
        return jp.clip(
            -12.0 * observation[..., :2] - 1.5 * observation[..., 2:4], -1.0, 1.0
        )

    def close(self):
        pass


register_environment("WorkbenchReach", WorkbenchReach, default_config)
