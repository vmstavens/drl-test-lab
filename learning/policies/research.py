"""Concrete DP and history-adapted RMA policies for the shared viewer/recorder."""

from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jp
import numpy as np

from learning.contracts import LearningContract
from learning.imitation.diffusion_policy import DPReference
from learning.reinforcement.drlr2 import Trainer
from learning.reinforcement.rma import adaptation_module, student_action
from learning.reinforcement.task import Task


def contract_from_metadata(value):
    value = dict(value)
    for name in (
        "observation_layout",
        "action_low",
        "action_high",
        "action_units",
        "privileged_layout",
    ):
        value[name] = tuple(value[name])
    result = LearningContract(**value)
    result.validate()
    return result


class DPPolicy:
    def __init__(self, checkpoint, config):
        self.contract = contract_from_metadata(config["contract"])
        self.reference = DPReference(
            checkpoint, self.contract, inference_steps=config["inference_steps"]
        )
        self.reset(seed=0)

    def reset(self, *, seed):
        self.key = jax.random.PRNGKey(seed)
        self.previous = None

    def act(self, observation):
        raw = jp.asarray(observation)[None]
        previous = raw if self.previous is None else self.previous
        self.key, key = jax.random.split(self.key)
        action = self.reference(
            raw,
            previous,
            jp.zeros((1, len(self.contract.action_low))),
            jp.zeros((1, 0)),
            key,
        )[0]
        self.previous = raw
        low, high = (
            np.asarray(self.contract.action_low),
            np.asarray(self.contract.action_high),
        )
        return low + (np.asarray(action) + 1) * 0.5 * (high - low)


class RMAPolicy:
    def __init__(self, checkpoint, config):
        if not config["accepted"]:
            raise ValueError(
                "Deployment bundle did not pass the student evaluation gate"
            )
        self.contract = contract_from_metadata(config["contract"])
        task = Task(
            config["stage"], domain_level=config["stage"]["task"]["env"]["domain_level"]
        )
        self.trainer = Trainer(
            config["stage"], task, None, center=config["center"], scale=config["scale"]
        )
        task.close()
        self.settings = config["rma"]
        self.adaptation = adaptation_module(self.trainer, self.settings)
        dummy = jp.zeros((1, self.trainer.width))
        actor = self.trainer.actor.init(jax.random.PRNGKey(0), dummy)
        self.frozen = flax.serialization.from_bytes(
            actor, Path(checkpoint).with_name("actor.flax").read_bytes()
        )
        history = jp.zeros(
            (1, self.settings["history_length"], self.trainer.o + self.trainer.a)
        )
        template = self.adaptation.init(jax.random.PRNGKey(0), history)
        self.params = flax.serialization.from_bytes(
            template, Path(checkpoint).read_bytes()
        )
        self.infer = jax.jit(
            lambda obs, previous, history: student_action(
                self.trainer,
                self.frozen,
                self.adaptation,
                self.params,
                obs,
                previous,
                history,
            )
        )
        self.reset(seed=0)

    def reset(self, *, seed):
        self.previous = np.zeros(self.trainer.a, np.float32)
        self.history = np.zeros(
            (self.settings["history_length"], self.trainer.o + self.trainer.a),
            np.float32,
        )
        self.steps = 0
        self.pending = None

    def act(self, observation):
        self.pending = np.asarray(observation, np.float32)
        obs = np.concatenate((self.pending, np.zeros(self.trainer.f), self.pending))[
            None
        ]
        action = np.asarray(self.infer(obs, self.previous[None], self.history[None]))[0]
        low, high = (
            np.asarray(self.contract.action_low),
            np.asarray(self.contract.action_high),
        )
        return low + (action + 1) * 0.5 * (high - low)

    def observe_transition(self, executed_action):
        self.previous = self.contract.normalize_action(executed_action).astype(
            np.float32
        )
        if self.steps % self.settings["sample_stride"] == 0:
            feature = np.concatenate(
                (
                    (self.pending - np.asarray(self.trainer.center))
                    / np.asarray(self.trainer.scale),
                    self.previous,
                )
            )
            self.history = np.concatenate((self.history[1:], feature[None]), axis=0)
        self.steps += 1


def load_dp_policy(*, checkpoint, config, deterministic):
    return DPPolicy(checkpoint, config)


def load_rma_policy(*, checkpoint, config, deterministic):
    return RMAPolicy(checkpoint, config)
