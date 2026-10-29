"""Versioned LEAP benchmarks using Playground's model and reset distribution.

Physics, actuator processing and observations live here, outside learning code.
The upstream moving-goal task and its reward are deliberately not modified.
"""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_hand import base, reorient
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts

from envs.registry import register_environment


def default_config():
    config = reorient.default_config()
    config.update(
        mode="reorient",
        episode_length=200,
        hold_seconds=0.25,
        success_threshold=0.2,
        mass_range=[1.0, 1.0],
        friction_range=[1.0, 1.0],
        rotation_speed=2.0,
        impulse_step=-1,
        impulse_velocity=[0.0, 0.0, 0.0],
        camera_offset=[0.0, 0.0, 0.0],
        lighting_scale=1.0,
        image_delay_steps=0,
        occlusion_fraction=0.0,
        menagerie_path=None,
    )
    return config


def rotation_config():
    config = default_config()
    config.mode = "rotate"
    return config


def orientation_error(quaternion, goal):
    # q and -q represent the same orientation. atan2 avoids acos' infinite slope.
    dot = jp.sum(quaternion * goal, axis=-1)
    sine = jp.sqrt(jp.maximum(1.0 - dot**2, 1e-8))
    return 2 * jp.arctan2(sine, jp.abs(dot) + 1e-8)


class SuperMimicHand(reorient.CubeReorient):
    """Fixed-goal reorientation or commanded z rotation, with dense rewards.

    Commands are increments in radians, clipped against actuator position limits.
    World state includes the accumulated targets; executed-action feedback is the
    realized increment, not the requested one. Physical failures end episodes;
    time limits are logged separately for correct value bootstrapping.
    """

    def __init__(self, config=None, config_overrides=None):
        mjx_env.MjxEnv.__init__(self, config or default_config(), config_overrides)
        if self._config.mode not in {"rotate", "reorient"}:
            raise ValueError("Hand mode must be rotate or reorient")
        if self._config.hold_seconds <= 0 or self._config.success_threshold <= 0:
            raise ValueError("Hold duration and success tolerance must be positive")
        for name in ("mass_range", "friction_range"):
            low, high = self._config[name]
            if not 0 < low <= high:
                raise ValueError(f"Invalid {name}")
        assets = base.get_assets()
        menagerie = self._config.menagerie_path
        path = Path(menagerie) if menagerie else Path(mjx_env.MENAGERIE_PATH)
        if not (path / "leap_hand" / "assets").is_dir():
            raise FileNotFoundError(
                "LEAP meshes are required. Fetch the pinned MuJoCo Menagerie "
                "with tools.fetch_research_assets and set task.env.menagerie_path."
            )
        mjx_env.update_assets(assets, path / "leap_hand" / "assets")
        self._model_assets = assets
        digest = hashlib.sha256(consts.CUBE_XML.read_bytes())
        for name, content in sorted(assets.items()):
            digest.update(name.encode())
            digest.update(content)
        self._asset_sha256 = digest.hexdigest()
        self._xml_path = str(consts.CUBE_XML)
        self._mj_model = mujoco.MjModel.from_xml_string(
            consts.CUBE_XML.read_text(), assets=assets
        )
        goal_body = self._mj_model.body("goal").id
        self._mj_model.geom_group[self._mj_model.geom_bodyid == goal_body] = 5
        self._mj_model.cam_pos[self._mj_model.camera("side").id] += np.asarray(
            self._config.camera_offset
        )
        self._mj_model.light_diffuse[:] *= self._config.lighting_scale
        self._mj_model.vis.headlight.diffuse[:] *= self._config.lighting_scale
        self._mj_model.vis.headlight.ambient[:] *= self._config.lighting_scale
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()
        self._renderer = None
        self._render_data = mujoco.MjData(self._mj_model)
        self._render_shape = None
        self._contract_name = (
            "SuperMimicCubeReorient"
            if self._config.mode == "reorient"
            else "SuperMimicCubeRotateZ"
        )
        if self._mj_model.nq != 23 or self._mj_model.nv != 22:
            raise ValueError("LEAP model layout changed; revise the versioned contract")

    @property
    def observation_layout(self):
        return [
            *[f"qpos_{i}" for i in range(23)],
            *[f"qvel_{i}" for i in range(22)],
            *self.model_spec["goal_layout"],
        ]

    @property
    def learning_contract(self):
        return dict(
            environment=self._contract_name,
            observation_layout=self.observation_layout,
            action_id="leap_executed_joint_increment_v1",
            action_low=[-self._config.action_scale] * 16,
            action_high=[self._config.action_scale] * 16,
            action_units=["rad"] * 16,
            control_dt=self.dt,
            sim_dt=self._config.sim_dt,
            coordinate_frame="playground_leap_world_wxyz_v1",
            privileged_layout=["cube_mass_scale", "cube_friction_scale"],
            preprocessing="leap_full_state_fixed_goal_v1",
            previous_action_source="executed",
        )

    @property
    def observation_size(self):
        return len(self.observation_layout)

    @property
    def render_spec(self):
        camera = self._mj_model.camera("side").id
        return dict(
            version=1,
            camera_position=self._mj_model.cam_pos[camera].tolist(),
            camera_quaternion=self._mj_model.cam_quat[camera].tolist(),
            lighting_scale=float(self._config.lighting_scale),
            occlusion_fraction=float(self._config.occlusion_fraction),
            target_ghost_visible=False,
        )

    @property
    def model_spec(self):
        goal = (
            [f"target_quat_{axis}" for axis in "wxyz"]
            if self._config.mode == "reorient"
            else ["target_z_angular_velocity_rad_s"]
        )
        return dict(
            version=1,
            asset_sha256=self._asset_sha256,
            controller="clipped_joint_increment_v1",
            actuator_lower=np.asarray(self._lowers).tolist(),
            actuator_upper=np.asarray(self._uppers).tolist(),
            state_layout=[
                *[f"qpos_{i}" for i in range(23)],
                *[f"qvel_{i}" for i in range(22)],
                *[f"motor_target_{i}_rad" for i in range(16)],
            ],
            goal_layout=goal,
            state_scale=[1.0] * 23 + [5.0] * 22 + [1.0] * 16,
        )

    def reset(self, rng):
        state = super().reset(rng)
        rng, mass_key, friction_key = jax.random.split(state.info["rng"], 3)
        factors = jp.stack(
            [
                jax.random.uniform(
                    mass_key,
                    (),
                    minval=self._config.mass_range[0],
                    maxval=self._config.mass_range[1],
                ),
                jax.random.uniform(
                    friction_key,
                    (),
                    minval=self._config.friction_range[0],
                    maxval=self._config.friction_range[1],
                ),
            ]
        )
        goal = (
            state.data.mocap_quat[0]
            if self._config.mode == "reorient"
            else jp.array([self._config.rotation_speed])
        )
        info = dict(
            privileged=factors,
            steps=jp.array(0),
            executed_action=jp.zeros(16),
            terminated=jp.array(False),
            goal=goal,
            hold=jp.array(0),
            rotation=jp.array(0.0),
            contact_active=jp.array(False),
            contact_change=jp.array(False),
            slip_proxy=jp.array(False),
            rng=rng,
        )
        result = state.replace(info=info, metrics=self._metrics(state.data, goal, 0, 0))
        return result.replace(
            obs=self.model_observation(self.model_state(result), goal, factors)
        )

    def _physical_model(self, factors):
        body_mass = self._mjx_model.body_mass.at[self._cube_body_id].multiply(
            factors[0]
        )
        body_inertia = self._mjx_model.body_inertia.at[self._cube_body_id].multiply(
            factors[0]
        )
        friction = self._mjx_model.geom_friction.at[self._cube_geom_id].multiply(
            factors[1]
        )
        return self._mjx_model.replace(
            body_mass=body_mass, body_inertia=body_inertia, geom_friction=friction
        )

    def step(self, state, action):
        action = jp.clip(action, -self._config.action_scale, self._config.action_scale)
        targets = jp.clip(state.data.ctrl + action, self._lowers, self._uppers)
        executed = targets - state.data.ctrl
        data = state.data
        impulse = jp.asarray(self._config.impulse_velocity)
        data = data.replace(
            qvel=data.qvel.at[16:19].add(
                jp.where(state.info["steps"] == self._config.impulse_step, impulse, 0.0)
            )
        )
        data = mjx_env.step(
            self._physical_model(state.info["privileged"]),
            data,
            targets,
            self.n_substeps,
        )
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        terminated = (data.qpos[18] < -0.05) | ~finite
        goal = state.info["goal"]
        error = (
            orientation_error(data.qpos[19:23], goal)
            if self._config.mode == "reorient"
            else jp.abs(data.qvel[21] - goal[0])
        )
        hold = jp.where(
            error < self._config.success_threshold, state.info["hold"] + 1, 0
        )
        rotation = state.info["rotation"] + data.qvel[21] * self.dt
        steps = state.info["steps"] + 1
        reward = (
            jp.exp(-(error**2) / 0.5)
            - 0.1 * jp.sum((data.qpos[16:19] - jp.array([0.1, 0.0, 0.05])) ** 2)
            - 0.001 * jp.sum((executed / self._config.action_scale) ** 2)
            - 0.0001 * jp.sum(data.qvel[:16] ** 2)
            - 5.0 * terminated
        ) * self.dt
        # This per-world proxy avoids Warp's shared packed contact arrays.
        contact_active = jp.linalg.norm(data.qfrc_constraint[16:22]) > 1e-6
        info = {
            **state.info,
            "steps": steps,
            "executed_action": executed,
            "terminated": terminated,
            "hold": hold,
            "rotation": rotation,
            "contact_active": contact_active,
            "contact_change": contact_active != state.info["contact_active"],
            "slip_proxy": contact_active & (jp.linalg.norm(data.qvel[16:19]) > 0.05),
        }
        following = state.replace(
            data=data,
            info=info,
            reward=reward,
            done=(terminated | (steps >= self._config.episode_length)).astype(
                jp.float32
            ),
            metrics=self._metrics(data, goal, hold, rotation),
        )
        return following.replace(
            obs=self.model_observation(
                self.model_state(following), goal, info["privileged"]
            )
        )

    def _metrics(self, data, goal, hold, rotation):
        error = (
            orientation_error(data.qpos[19:23], goal)
            if self._config.mode == "reorient"
            else jp.abs(data.qvel[21] - goal[0])
        )
        success = hold * self.dt >= self._config.hold_seconds
        if self._config.mode == "rotate":
            success = rotation * jp.sign(goal[0]) >= 2 * jp.pi
        return dict(
            is_success=jp.asarray(success, dtype=jp.float32),
            orientation_or_velocity_error=error,
            dropped=(data.qpos[18] < -0.05).astype(jp.float32),
            rotation_rad=jp.asarray(rotation, dtype=jp.float32),
        )

    def model_state(self, state):
        return jp.concatenate((state.data.qpos, state.data.qvel, state.data.ctrl))

    def model_goal(self, state):
        return state.info["goal"]

    def model_observation(self, state, goal, context):
        del context
        return jp.concatenate((state[:45], goal))

    def model_project(self, state):
        quaternion = state[19:23]
        norm = jp.sqrt(jp.maximum(jp.sum(quaternion**2), 1e-12))
        quaternion = jp.where(
            norm > 1e-6,
            quaternion / jp.maximum(norm, 1e-6),
            jp.array([1.0, 0.0, 0.0, 0.0]),
        )
        return (
            state.at[19:23]
            .set(quaternion)
            .at[45:]
            .set(jp.clip(state[45:], self._lowers, self._uppers))
        )

    def visual_proprioception(self, state, goal):
        return jp.concatenate((state[..., :16], state[..., 23:39], goal), axis=-1)

    def model_action(self, state, requested):
        scale = self._config.action_scale
        targets = jp.clip(
            state[45:] + jp.clip(requested, -1, 1) * scale, self._lowers, self._uppers
        )
        return (targets - state[45:]) / scale

    def model_transition(self, predicted, state, executed):
        return predicted.at[45:].set(state[45:] + executed * self._config.action_scale)

    def pixel_observation(self, state, *, height, width, camera="side"):
        """Host renderer for visual experiments; rendering cost is measured."""
        if self._render_shape != (height, width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
            self._render_shape = (height, width)
        self._render_data.qpos[:] = np.asarray(state.data.qpos)
        self._render_data.qvel[:] = np.asarray(state.data.qvel)
        self._render_data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
        self._render_data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
        mujoco.mj_forward(self._mj_model, self._render_data)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0  # Collision overlays, including the untextured cube.
        option.geomgroup[5] = 0  # Target ghost; retain the real cube's marked surface.
        self._renderer.update_scene(
            self._render_data, camera=camera, scene_option=option
        )
        image = self._renderer.render().copy()
        fraction = self._config.occlusion_fraction
        if not 0 <= fraction < 1:
            raise ValueError("Occlusion fraction must be in [0,1)")
        radius = int(min(height, width) * fraction / 2)
        if radius:
            image[
                height // 2 - radius : height // 2 + radius,
                width // 2 - radius : width // 2 + radius,
            ] = 0
        return image

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


register_environment("SuperMimicCubeReorient", SuperMimicHand, default_config)
register_environment("SuperMimicCubeRotateZ", SuperMimicHand, rotation_config)
