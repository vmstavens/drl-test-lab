"""Portable actor exports with strict physical, model and visual contracts."""

import hashlib
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jp
import numpy as np

from learning.supermimic.networks import Actor


class SuperMimicPolicy:
    def __init__(self, actor, parameters, config):
        self.actor, self.parameters, self.config = actor, parameters, config
        self.visual = config["observation"] == "rgb"
        self.observation_mode = config["observation"]
        self.pixel_options = config["pixels"]
        self.contract = config["contract"]
        self.scale = jp.asarray(config["observation_scale"])
        self.action_low = np.asarray(self.contract["action_low"])
        self.action_high = np.asarray(self.contract["action_high"])
        self._inference = jax.jit(
            lambda features, pixels: jp.tanh(
                actor.apply(parameters, features, pixels)[0]
            )
        )
        self.reset(seed=0)

    def reset(self, *, seed):
        self.previous = np.zeros(len(self.action_low), dtype=np.float32)
        self.pixel_queue = []

    def observation_from_state(self, environment, state):
        return policy_observation(environment, state, self)

    def batch_action(self, observation, previous):
        if self.visual:
            raise ValueError("State teacher interface cannot consume a visual actor")
        features = jp.concatenate(
            (jp.asarray(observation) / self.scale, previous), axis=-1
        )
        return self._inference(features, None)

    def act(self, observation):
        if self.visual:
            # Only joint proprioception and the declared goal reach the actor.
            proprio = jp.asarray(observation["proprioception"])
            goal_size = len(self.config["model_spec"]["goal_layout"])
            scale = jp.concatenate((jp.ones(16), jp.full(16, 5.0), jp.ones(goal_size)))
            features = jp.concatenate((proprio / scale, jp.asarray(self.previous)))[
                None
            ]
            frame = np.asarray(observation["pixels"])
            if frame.shape != tuple(self.pixel_options["shape"]):
                raise ValueError(
                    "Image dimensions differ from the exported visual contract"
                )
            delay = observation.get("delay_steps", 0)
            if type(delay) is not int or delay < 0:
                raise ValueError("Image delay must be a nonnegative integer")
            self.pixel_queue.append(frame.copy())
            self.pixel_queue = self.pixel_queue[-(delay + 1) :]
            pixels = jp.asarray(self.pixel_queue[0], dtype=jp.float32)[None] / 255.0
            normalized = np.asarray(self._inference(features, pixels)[0])
        else:
            normalized = np.asarray(
                self.batch_action(
                    jp.asarray(observation)[None], jp.asarray(self.previous)[None]
                )[0]
            )
        return self.action_low + (normalized + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    def observe_transition(self, executed_action):
        self.previous = np.asarray(
            2
            * (executed_action - self.action_low)
            / (self.action_high - self.action_low)
            - 1,
            dtype=np.float32,
        )


def export_policy(trainer, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = flax.serialization.to_bytes(trainer.parameters.actor)
    (directory / "actor.msgpack").write_bytes(payload)
    # World model/critic are research artifacts, never required for deployment.
    training_payload = flax.serialization.to_bytes(
        dict(
            model=trainer.parameters.model,
            critic=trainer.parameters.critic,
            pixels=trainer.parameters.pixels,
        )
    )
    (directory / "training.msgpack").write_bytes(training_payload)
    from learning.supermimic.artifacts import model_contract

    (directory / "world_model.json").write_text(
        json.dumps(
            dict(
                contract=model_contract(trainer),
                checkpoint="training.msgpack",
                sha256=hashlib.sha256(training_payload).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    options = trainer.options
    config = dict(
        contract=trainer.task.contract.to_dict(),
        model_spec=trainer.spec,
        observation=options["observation"],
        observation_scale=np.asarray(trainer.obs_scale).tolist(),
        actor_widths=options["actor_widths"],
        initial_std=options["action_noise"],
        pixels=options["pixels"],
        render_spec=trainer.task.base.render_spec if trainer.visual else None,
        actor_sha256=hashlib.sha256(payload).hexdigest(),
        training_seed=trainer.config["seed"],
        method=trainer.method,
        feature_contract="current_observation_then_executed_previous_v1",
    )
    manifest = dict(
        schema_version=1,
        policy_type="supermimic_actor_v1",
        loader="learning.supermimic.policy:load_actor",
        checkpoint="actor.msgpack",
        action_semantics=trainer.task.contract.action_id,
        observation_dim=trainer.task.observation_size,
        action_dim=trainer.task.action_size,
        config=config,
    )
    (directory / "policy.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory / "policy.json"


def load_actor(*, checkpoint, config, deterministic=True):
    if (
        config.get("feature_contract")
        != "current_observation_then_executed_previous_v1"
    ):
        raise ValueError("Unsupported actor history contract")
    if not deterministic:
        raise ValueError(
            "SuperMimic deployment exports support deterministic inference"
        )
    payload = Path(checkpoint).read_bytes()
    if hashlib.sha256(payload).hexdigest() != config["actor_sha256"]:
        raise ValueError("Actor checkpoint hash differs from manifest")
    visual = config["observation"] == "rgb"
    if config["observation"] not in {"state", "rgb"}:
        raise ValueError("Unsupported observation contract")
    action_size = len(config["contract"]["action_low"])
    goal_size = len(config["model_spec"]["goal_layout"])
    features = (
        32 + goal_size if visual else len(config["observation_scale"])
    ) + action_size
    actor = Actor(
        action_size, tuple(config["actor_widths"]), visual, config["initial_std"]
    )
    pixels = jp.zeros((1, *config["pixels"]["shape"])) if visual else None
    template = actor.init(jax.random.PRNGKey(0), jp.zeros((1, features)), pixels)
    parameters = flax.serialization.from_bytes(template, payload)
    return SuperMimicPolicy(actor, parameters, config)


def policy_observation(environment, state, policy):
    """Render only for a policy explicitly declaring the visual input contract."""
    if getattr(policy, "observation_mode", "state") != "rgb":
        return state.obs
    options = policy.pixel_options
    return dict(
        pixels=environment.pixel_observation(
            state,
            height=options["shape"][0],
            width=options["shape"][1],
            camera=options["camera"],
        ),
        proprioception=environment.visual_proprioception(
            environment.model_state(state), environment.model_goal(state)
        ),
        delay_steps=int(environment._config.image_delay_steps),
    )
