"""Dense research variant; the canonical sparse WorkbenchReach is unchanged."""

import jax.numpy as jp

from envs.registry import register_environment
from envs.workbench_reach import WorkbenchReach, default_config


def dense_reward(observation, action):
    return (
        jp.exp(-jp.sum(observation[..., :2] ** 2, axis=-1) / 0.01)
        - 0.01 * jp.sum(observation[..., 2:] ** 2, axis=-1)
        - 0.001 * jp.sum(action**2, axis=-1)
    )


class SuperMimicReach(WorkbenchReach):
    """Fixed-duration dense reaching with a privileged deterministic model state."""

    @property
    def learning_contract(self):
        return {
            **super().learning_contract,
            "environment": "SuperMimicReach",
            "preprocessing": "supermimic_dense_reach_v1",
        }

    @property
    def model_spec(self):
        return dict(
            version=1,
            state_layout=list(self.observation_layout),
            goal_layout=["target_x_m", "target_y_m"],
            state_scale=[0.2, 0.2, 1.0, 1.0],
        )

    def step(self, state, action):
        following = super().step(state, action)
        finite = jp.all(jp.isfinite(following.obs))
        terminal = ~finite
        timeout = following.info["steps"] >= self._config.episode_length
        return following.replace(
            reward=dense_reward(following.obs, following.info["executed_action"]),
            done=(terminal | timeout).astype(jp.float32),
            info={**following.info, "terminated": terminal},
        )

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(info={**state.info, "terminated": jp.array(False)})

    def model_state(self, state):
        return state.obs

    def model_goal(self, state):
        return jp.zeros(2)

    def model_observation(self, state, goal, context):
        del context
        return state.at[:2].add(-goal)

    def model_project(self, state):
        return state


register_environment("SuperMimicReach", SuperMimicReach, default_config)
