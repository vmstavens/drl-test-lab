"""The sole task boundary used by collection, training and evaluation."""

import copy

import jax
import jax.numpy as jp

from learning.contracts import LearningContract
from learning.evaluation.rollouts import create_environment


class Task:
    """Batch raw environment state; expose normalized commands and privileged factors.

    Custom environments provide info['privileged'] and info['executed_action'].
    The latter is in physical command units, after controller clipping/filtering.
    There is no automatic reset: the collector owns episode boundaries.
    """

    def __init__(self, config, *, domain_level=None):
        self.contract = LearningContract.from_config(config)
        cfg = copy.deepcopy(config)
        if domain_level is not None:
            cfg["task"]["env"]["domain_level"] = domain_level
        self.base = create_environment(cfg)
        self.contract.require(self.base.learning_contract)
        self.reset = jax.jit(jax.vmap(self._reset))
        self.step = jax.jit(jax.vmap(self._step))
        self.action_size = len(self.contract.action_low)
        self.observation_size = len(self.contract.observation_layout)

    def _reset(self, key):
        state = self.base.reset(key)
        return state.replace(
            obs=jp.concatenate((state.obs, state.info["privileged"], state.obs)),
            info={
                **state.info,
                "executed_normalized_action": jp.zeros_like(
                    state.info["executed_action"]
                ),
            },
        )

    def _step(self, state, normalized):
        low, high = (
            jp.asarray(self.contract.action_low),
            jp.asarray(self.contract.action_high),
        )
        physical = low + (jp.clip(normalized, -1, 1) + 1) * 0.5 * (high - low)
        following = self.base.step(state, physical)
        executed = self.executed(following)
        return following.replace(
            obs=jp.concatenate(
                (
                    following.obs,
                    following.info["privileged"],
                    state.obs[: self.observation_size],
                )
            ),
            info={**following.info, "executed_normalized_action": executed},
        )

    def executed(self, state):
        action = state.info["executed_action"]
        low, high = (
            jp.asarray(self.contract.action_low),
            jp.asarray(self.contract.action_high),
        )
        return 2 * (action - low) / (high - low) - 1

    def close(self):
        self.base.close()
