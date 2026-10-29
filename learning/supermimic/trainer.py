"""Online learned dynamics, analytic policy gradients and matched PPO baselines.

This is an independent method, not a replacement for the canonical DRLR recipe.
The environment Task remains the only simulation boundary.
"""

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import struct

from learning.evaluation.rollouts import EpisodeResult, write_results
from learning.reinforcement.task import Task
from learning.supermimic.compilation import CompiledKernel
from learning.supermimic.networks import MLP, Actor, PixelModel, WorldModel
from learning.supermimic.objectives import (
    action_log_prob,
    lambda_returns,
    survival_return,
)
from learning.supermimic.replay import SequenceReplay


@struct.dataclass
class Parameters:
    actor: object
    model: object
    critic: object
    pixels: object
    actor_opt: object
    model_opt: object
    critic_opt: object
    pixels_opt: object
    rng: object


def select_reset(following, fresh, done):
    def select(old, new):
        return jp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old)

    # Warp has shared contact/solver arrays without an environment axis.
    # Data.where preserves these buffers; apply its mask inside the vmap.
    data = jax.vmap(lambda old, new, mask: old.where(mask, new))(
        following.data, fresh.data, done
    )
    result = jax.tree.map(
        select, following.replace(data=None), fresh.replace(data=None)
    )
    return result.replace(data=data)


def validate_config(config):
    options = config["supermimic"]
    if options["method"] not in {"supermimic", "ppo_sim", "ppo_wm", "visual_dagger"}:
        raise ValueError("Unknown SuperMimic comparison method")
    if options["freeze_model"] and (
        options["method"] not in {"supermimic", "ppo_wm"}
        or (not options["model_manifest"] and not config["runtime"]["dry_run"])
    ):
        raise ValueError(
            "Frozen model comparisons require a shared world-model manifest"
        )
    positive = (
        "iterations",
        "num_envs",
        "replay_steps",
        "window_steps",
        "horizon_steps",
        "collect_steps",
        "batch_size",
        "model_updates",
        "critic_epochs",
        "ppo_epochs",
        "evaluation_interval",
        "checkpoint_interval",
        "learning_rate",
        "reward_scale",
    )
    for name in positive:
        if not math.isfinite(options[name]) or options[name] <= 0:
            raise ValueError(f"supermimic.{name} must be positive")
        if (
            name not in {"learning_rate", "reward_scale"}
            and type(options[name]) is not int
        ):
            raise ValueError(f"supermimic.{name} must be an integer")
    for name in ("actor_widths", "model_widths", "critic_widths", "pixel_widths"):
        if not options[name] or any(type(x) is not int or x < 1 for x in options[name]):
            raise ValueError(f"supermimic.{name} must contain positive integer widths")
    if not 0 < options["holdout_envs"] < options["num_envs"]:
        raise ValueError("Reserve at least one disjoint holdout environment")
    if options["replay_steps"] < options["window_steps"]:
        raise ValueError("Replay cannot hold a model-fitting window")
    if config["task"]["env"]["episode_length"] < options["window_steps"]:
        raise ValueError("Model windows must fit within one episode")
    if not 0 < options["gamma"] <= 1 or not 0 <= options["gae_lambda"] <= 1:
        raise ValueError("Invalid discount or GAE lambda")
    if options["observation"] not in {"state", "rgb"}:
        raise ValueError("Observation must be state or rgb")
    if options["method"] == "visual_dagger":
        if options["observation"] != "rgb" or (
            not options["teacher_manifest"]
            and not config.get("runtime", {}).get("dry_run", False)
        ):
            raise ValueError("Visual DAgger requires RGB and a state teacher manifest")
    if options["method"] in {"ppo_sim", "ppo_wm"} and not options["use_critic"]:
        raise ValueError(
            "The no-critic ablation is supported for analytic updates only"
        )
    if options["observation"] == "rgb" and config["task"]["name"] == "supermimic_reach":
        raise ValueError("RGB experiments require a hand benchmark")
    if config["task"]["env"].get("image_delay_steps", 0):
        raise ValueError(
            "Image delay is an evaluation condition; training needs a Markov image head"
        )
    if not math.isfinite(options["action_noise"]) or options["action_noise"] <= 0:
        raise ValueError("Action standard deviation must be positive")
    if (
        options["diagnostics"]["probe_steps"] < 1
        or options["diagnostics"]["probe_epsilon"] <= 0
    ):
        raise ValueError("Gradient probes need a positive horizon and epsilon")
    evaluation = options["evaluation"]
    if evaluation["episodes"] < 1 or evaluation["seed_start"] < 0:
        raise ValueError("Evaluation requires a positive episode count and seed range")
    if (
        evaluation["seed_start"]
        <= config["seed"]
        < evaluation["seed_start"] + evaluation["episodes"]
    ):
        raise ValueError("Training seed overlaps evaluation episode seeds")
    if not 0 < evaluation["target_success_rate"] <= 1:
        raise ValueError("Invalid target success rate")


class Trainer:
    def __init__(self, config, task):
        validate_config(config)
        self.config, self.task = config, task
        self.options = config["supermimic"]
        self.n = self.options["num_envs"]
        self.train_n = self.n - self.options["holdout_envs"]
        self.method = self.options["method"]
        self.visual = self.options["observation"] == "rgb"
        self.spec = task.base.model_spec
        self.state_size = len(self.spec["state_layout"])
        self.goal_size = len(self.spec["goal_layout"])
        self.scale = jp.asarray(self.spec["state_scale"], dtype=jp.float32)
        if self.scale.shape != (self.state_size,) or not np.all(
            np.asarray(self.scale) > 0
        ):
            raise ValueError(
                "World-state scales must be positive and match the state layout"
            )
        self.reward_scale = self.options["reward_scale"]
        self.pixel_shape = tuple(self.options["pixels"]["shape"])
        if (
            len(self.pixel_shape) != 3
            or self.pixel_shape[-1] != 3
            or min(self.pixel_shape) < 1
        ):
            raise ValueError("Pixel shape must be positive H,W,3")
        self.actor = Actor(
            task.action_size,
            tuple(self.options["actor_widths"]),
            self.visual,
            self.options["action_noise"],
        )
        self.model = WorldModel(self.state_size, tuple(self.options["model_widths"]))
        self.critic = MLP(tuple(self.options["critic_widths"]), 1)
        self.pixel_model = PixelModel(
            self.pixel_shape, tuple(self.options["pixel_widths"])
        )
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.clip(0.1),
            optax.adam(self.options["learning_rate"], b1=0.9, b2=0.95),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(self.options["learning_rate"])
        )
        self.replay = SequenceReplay(
            self.options["replay_steps"], self.n, self.options["holdout_envs"]
        )
        self.numpy_rng = np.random.default_rng(config["seed"])
        key = jax.random.fold_in(jax.random.PRNGKey(config["seed"]), 0)
        key, reset_key = jax.random.split(key)
        self.environment = task.reset(jax.random.split(reset_key, self.n))
        self.previous = jp.zeros((self.n, task.action_size))
        state, goal, context = self.describe(self.environment)
        self.obs_scale = jp.ones(task.observation_size)
        # Explicit physical scales, frozen for the entire run and export.
        if task.observation_size == self.state_size:
            self.obs_scale = self.scale
        elif task.observation_size == 45 + self.goal_size:
            self.obs_scale = jp.concatenate((self.scale[:45], jp.ones(self.goal_size)))
        features = self.features(state, goal, context, self.previous)
        keys = jax.random.split(key, 5)
        dummy_pixels = jp.zeros((self.n, *self.pixel_shape)) if self.visual else None
        actor = self.actor.init(keys[0], features, dummy_pixels)
        model = self.model.init(
            keys[1],
            state / self.scale,
            self.previous,
            context,
            goal,
            method=self.model.initialize,
        )
        critic = self.critic.init(
            keys[2], self.value_features(state, goal, context, self.previous)
        )
        pixels = (
            self.pixel_model.init(
                keys[3], state / self.scale, jp.concatenate((context, goal), axis=-1)
            )
            if self.visual
            else {}
        )
        self.parameters = Parameters(
            actor,
            model,
            critic,
            pixels,
            self.actor_optimizer.init(actor),
            self.optimizer.init(model),
            self.optimizer.init(critic),
            self.optimizer.init(pixels),
            keys[4],
        )
        self._collect = CompiledKernel(self.collect_step)
        if self.options["model_manifest"]:
            from learning.supermimic.artifacts import load_world_model

            self.parameters = load_world_model(self, self.options["model_manifest"])
        self._model_update = CompiledKernel(self.model_update)
        self._analytic_update = CompiledKernel(self.analytic_update)
        self._ppo_update = CompiledKernel(self.ppo_update)
        self._critic_update = CompiledKernel(self.critic_update)
        self._imagine = CompiledKernel(self.imagine)
        self._dagger_update = CompiledKernel(self.dagger_update)
        self._mean_action = CompiledKernel(self.mean_action)
        self.teacher = None
        if self.method == "visual_dagger":
            from learning.policies.checkpoints import load_policy

            self.teacher = load_policy(
                Path(self.options["teacher_manifest"]), task.base
            )
            if self.teacher.visual:
                raise ValueError("DAgger teacher must use state observations")

    def describe(self, environment):
        return (
            self.task.model_state(environment),
            self.task.model_goal(environment),
            self.task.model_context(environment),
        )

    def features(self, state, goal, context, previous):
        if self.visual:
            observation = self.task.base.visual_proprioception(state, goal)
            # Hand joints in radians, velocities in units of 5 rad/s.
            scale = jp.concatenate(
                (jp.ones(16), jp.full(16, 5.0), jp.ones(self.goal_size))
            )
            observation = observation / scale
        else:
            observation = (
                self.task.model_observation(state, goal, context) / self.obs_scale
            )
        return jp.concatenate((observation, previous), axis=-1)

    def value_features(self, state, goal, context, previous):
        return jp.concatenate((state / self.scale, goal, context, previous), axis=-1)

    def value(self, params, state, goal, context, previous):
        return self.critic.apply(
            params, self.value_features(state, goal, context, previous)
        )[..., 0]

    def predicted_pixels(self, params, state, goal, context):
        if not self.visual:
            return None
        return self.pixel_model.apply(
            params, state / self.scale, jp.concatenate((context, goal), axis=-1)
        )

    def mean_action(self, actor, state, goal, context, previous, pixels):
        mean, _ = self.actor.apply(
            actor, self.features(state, goal, context, previous), pixels
        )
        return jp.tanh(mean)

    def render_batch(self, environment):
        if not self.visual:
            return np.zeros((self.n, 0), dtype=np.float32)
        frames = []
        for i in range(environment.done.shape[0]):
            item = jax.tree.map(
                lambda x, index=i: x[index], environment.replace(data=None)
            ).replace(data=environment.data[i])
            frames.append(
                self.task.base.pixel_observation(
                    item,
                    height=self.pixel_shape[0],
                    width=self.pixel_shape[1],
                    camera=self.options["pixels"]["camera"],
                )
            )
        return np.asarray(frames, dtype=np.float32) / 255.0

    def collect_step(
        self,
        parameters,
        environment,
        previous,
        pixels,
        requested_override,
        use_override,
    ):
        key, noise_key, reset_key = jax.random.split(parameters.rng, 3)
        state, goal, context = self.describe(environment)
        features = self.features(state, goal, context, previous)
        mean, log_std = self.actor.apply(
            parameters.actor, features, pixels if self.visual else None
        )
        raw = mean + jp.exp(log_std) * jax.random.normal(noise_key, mean.shape)
        # Analytic-gradient policies use deterministic actor + external action noise.
        if self.method in {"supermimic", "visual_dagger"}:
            requested = jp.clip(
                jp.tanh(mean)
                + self.options["action_noise"]
                * jax.random.normal(noise_key, mean.shape),
                -1,
                1,
            )
        else:
            requested = jp.tanh(raw)
        requested = jp.where(use_override, requested_override, requested)
        following = self.task.step(environment, requested)
        executed = self.task.executed(following)
        done = following.done.astype(bool)
        next_previous = jp.where(done[:, None], 0.0, executed)
        next_state, _, _ = self.describe(following)
        transition = dict(
            state=state,
            goal=goal,
            context=context,
            previous=previous,
            action=executed,
            requested=requested,
            raw=raw,
            log_probability=action_log_prob(mean, log_std, raw),
            next_state=next_state,
            next_previous=next_previous,
            reward=following.reward / self.reward_scale,
            terminal=following.info["terminated"].astype(jp.float32),
            done=done,
            pixels=pixels,
            contact_change=following.info.get(
                "contact_change", jp.zeros(self.n, dtype=bool)
            ),
            slip_proxy=following.info.get("slip_proxy", jp.zeros(self.n, dtype=bool)),
            dropped=following.metrics.get("dropped", jp.zeros(self.n)),
        )
        fresh = self.task.reset(jax.random.split(reset_key, self.n))
        environment = select_reset(following, fresh, done)
        return parameters.replace(rng=key), environment, next_previous, transition

    def predict(self, model, state, action, goal, context):
        predicted = (
            self.model.apply(model, state / self.scale, action, context) * self.scale
        )
        following = self.task.model_transition(predicted, state, action)
        reward, logit = self.model.apply(
            model,
            state / self.scale,
            action,
            following / self.scale,
            goal,
            context,
            method=self.model.outcomes,
        )
        return following, reward, logit

    def model_loss(self, model, batch):
        def body(state, transition):
            following, reward, logit = self.predict(
                model,
                state,
                transition["action"],
                transition["goal"],
                transition["context"],
            )
            error = jp.mean(((following - transition["next_state"]) / self.scale) ** 2)
            reward_loss = jp.mean((reward - transition["reward"]) ** 2)
            terminal_loss = jp.mean(
                optax.sigmoid_binary_cross_entropy(logit, transition["terminal"])
            )
            return following, jp.stack((error, reward_loss, terminal_loss))

        time_major = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), batch)
        _, losses = jax.lax.scan(body, batch["state"][:, 0], time_major)
        losses = jp.mean(losses, axis=0)
        return jp.sum(losses), losses

    def model_update(self, parameters, batch):
        (loss, parts), gradient = jax.value_and_grad(self.model_loss, has_aux=True)(
            parameters.model, batch
        )
        updates, optimizer = self.optimizer.update(
            gradient, parameters.model_opt, parameters.model
        )
        model = optax.apply_updates(parameters.model, updates)
        pixels, pixels_opt = parameters.pixels, parameters.pixels_opt
        pixel_loss = jp.array(0.0)
        if self.visual:

            def objective(params):
                state = batch["state"].reshape((-1, self.state_size))
                goal = batch["goal"].reshape((-1, self.goal_size))
                context = batch["context"].reshape((-1, batch["context"].shape[-1]))
                predicted = self.predicted_pixels(params, state, goal, context)
                return jp.mean(
                    (predicted - batch["pixels"].reshape(predicted.shape)) ** 2
                )

            pixel_loss, gradient = jax.value_and_grad(objective)(pixels)
            updates, pixels_opt = self.optimizer.update(gradient, pixels_opt, pixels)
            pixels = optax.apply_updates(pixels, updates)
        return parameters.replace(
            model=model, model_opt=optimizer, pixels=pixels, pixels_opt=pixels_opt
        ), dict(
            model_loss=loss,
            state_loss=parts[0],
            reward_loss=parts[1],
            terminal_loss=parts[2],
            pixel_loss=pixel_loss,
        )

    def imagine(self, parameters, starts, actor=None):
        actor = parameters.actor if actor is None else actor
        model = jax.lax.stop_gradient(parameters.model)
        pixel_params = jax.lax.stop_gradient(parameters.pixels)
        goal, context = starts["goal"], starts["context"]

        def body(carry, key):
            state, previous = carry
            pixels = self.predicted_pixels(pixel_params, state, goal, context)
            features = self.features(state, goal, context, previous)
            mean, log_std = self.actor.apply(actor, features, pixels)
            raw = mean
            if self.method == "ppo_wm":
                raw = raw + jp.exp(log_std) * jax.random.normal(key, mean.shape)
            requested = jp.tanh(raw)
            action = self.task.model_action(state, requested)
            following, reward, logit = self.predict(model, state, action, goal, context)
            terminal = jax.nn.sigmoid(logit)
            # Soft survival retains differentiability; there is no imagined reset.
            transition = dict(
                state=state,
                previous=previous,
                action=action,
                goal=goal,
                context=context,
                next_state=following,
                next_previous=action,
                reward=reward,
                terminal=terminal,
                done=terminal,
                pixels=pixels if self.visual else jp.zeros((state.shape[0], 0)),
                raw=raw,
                log_probability=action_log_prob(mean, log_std, raw),
            )
            return (following, action), transition

        keys = jax.random.split(parameters.rng, self.options["horizon_steps"])
        (final_state, final_previous), rollout = jax.lax.scan(
            body, (starts["state"], starts["previous"]), keys
        )
        return rollout, final_state, final_previous

    def analytic_update(self, parameters, starts):
        def objective(actor):
            rollout, final_state, final_previous = self.imagine(
                parameters, starts, actor
            )
            final_value = self.value(
                jax.lax.stop_gradient(parameters.critic),
                final_state,
                starts["goal"],
                starts["context"],
                final_previous,
            )
            if not self.options["use_critic"]:
                final_value = jp.zeros_like(final_value)
            returns = survival_return(
                rollout["reward"],
                rollout["terminal"],
                final_value,
                self.options["gamma"],
            )
            return -jp.mean(returns), rollout

        (loss, rollout), gradient = jax.value_and_grad(objective, has_aux=True)(
            parameters.actor
        )
        updates, optimizer = self.actor_optimizer.update(
            gradient, parameters.actor_opt, parameters.actor
        )
        return (
            parameters.replace(
                actor=optax.apply_updates(parameters.actor, updates),
                actor_opt=optimizer,
            ),
            loss,
            rollout,
        )

    def critic_targets(self, parameters, rollout, *, simulator=False):
        value = self.value(
            parameters.critic,
            rollout["state"],
            rollout["goal"],
            rollout["context"],
            rollout["previous"],
        )
        following = self.value(
            parameters.critic,
            rollout["next_state"],
            rollout["goal"],
            rollout["context"],
            rollout["next_previous"],
        )
        if simulator:
            # A timeout bootstraps its final state, but cannot propagate advantages
            # from the next reset episode backwards into the previous episode.
            delta = (
                rollout["reward"]
                + self.options["gamma"] * (1 - rollout["terminal"]) * following
                - value
            )

            def body(carry, inputs):
                delta, terminal, done = inputs
                advantage = (
                    delta
                    + self.options["gamma"]
                    * self.options["gae_lambda"]
                    * (1 - terminal)
                    * (1 - done)
                    * carry
                )
                return advantage, advantage

            _, advantage = jax.lax.scan(
                body,
                jp.zeros_like(value[-1]),
                (delta, rollout["terminal"], rollout["done"]),
                reverse=True,
            )
            targets = advantage + value
        else:
            values = jp.concatenate((value, following[-1:]), axis=0)
            targets = lambda_returns(
                rollout["reward"],
                rollout["terminal"],
                values,
                self.options["gamma"],
                self.options["gae_lambda"],
            )
            advantage = targets - value
        return jax.lax.stop_gradient(targets), jax.lax.stop_gradient(advantage)

    def critic_update(self, parameters, batch, targets):
        def objective(critic):
            values = self.value(
                critic,
                batch["state"],
                batch["goal"],
                batch["context"],
                batch["previous"],
            )
            return jp.mean((values - targets) ** 2)

        loss, gradient = jax.value_and_grad(objective)(parameters.critic)
        updates, optimizer = self.optimizer.update(
            gradient, parameters.critic_opt, parameters.critic
        )
        return parameters.replace(
            critic=optax.apply_updates(parameters.critic, updates), critic_opt=optimizer
        ), loss

    def ppo_update(self, parameters, batch, advantage, weights):
        def objective(actor):
            features = self.features(
                batch["state"], batch["goal"], batch["context"], batch["previous"]
            )
            mean, log_std = self.actor.apply(
                actor, features, batch["pixels"] if self.visual else None
            )
            log_probability = action_log_prob(mean, log_std, batch["raw"])
            ratio = jp.exp(log_probability - batch["log_probability"])
            clipped = jp.clip(
                ratio, 1 - self.options["ppo_clip"], 1 + self.options["ppo_clip"]
            )
            surrogate = jp.minimum(ratio * advantage, clipped * advantage)
            entropy = jp.sum(log_std + 0.5 * (1 + jp.log(2 * jp.pi)), axis=-1)
            return -jp.sum(
                weights * (surrogate + self.options["entropy_coefficient"] * entropy)
            ) / jp.maximum(jp.sum(weights), 1e-8)

        loss, gradient = jax.value_and_grad(objective)(parameters.actor)
        updates, optimizer = self.actor_optimizer.update(
            gradient, parameters.actor_opt, parameters.actor
        )
        return parameters.replace(
            actor=optax.apply_updates(parameters.actor, updates), actor_opt=optimizer
        ), loss

    def dagger_update(self, parameters, batch):
        def objective(actor):
            predicted = self.mean_action(
                actor,
                batch["state"],
                batch["goal"],
                batch["context"],
                batch["previous"],
                batch["pixels"],
            )
            return jp.mean((predicted - batch["teacher_action"]) ** 2)

        loss, gradient = jax.value_and_grad(objective)(parameters.actor)
        updates, optimizer = self.actor_optimizer.update(
            gradient, parameters.actor_opt, parameters.actor
        )
        return parameters.replace(
            actor=optax.apply_updates(parameters.actor, updates), actor_opt=optimizer
        ), loss

    def sample(self, length=1, *, holdout=False):
        return jax.tree.map(
            jp.asarray,
            self.replay.sample(
                self.numpy_rng, self.options["batch_size"], length, holdout=holdout
            ),
        )

    def collect(self, steps, iteration):
        transitions = []
        for _ in range(steps):
            pixels = self.render_batch(self.environment)
            teacher_action = jp.zeros_like(self.previous)
            use_override = jp.array(False)
            if self.teacher is not None:
                state, goal, context = self.describe(self.environment)
                observation = self.task.model_observation(state, goal, context)
                teacher_action = self.teacher.batch_action(observation, self.previous)
                probability = max(
                    0.0,
                    1 - iteration / max(1, self.options["dagger_teacher_iterations"]),
                )
                use_override = jp.asarray(
                    self.numpy_rng.random((self.n, 1)) < probability
                )
            self.parameters, self.environment, self.previous, transition = (
                self._collect(
                    self.parameters,
                    self.environment,
                    self.previous,
                    jp.asarray(pixels),
                    teacher_action,
                    use_override,
                )
            )
            if self.teacher is not None:
                transition["teacher_action"] = teacher_action
            # Materialization synchronizes simulation; its cost belongs to collection.
            transition = jax.tree.map(np.asarray, transition)
            self.replay.add(transition)
            transitions.append(
                jax.tree.map(lambda x: jp.asarray(x[: self.train_n]), transition)
            )
        return jax.tree.map(lambda *xs: jp.stack(xs), *transitions)

    def train_iteration(self, iteration):
        rollout = self.collect(self.options["collect_steps"], iteration)
        metrics = {}
        if self.method in {"supermimic", "ppo_wm"}:
            if not self.options["freeze_model"]:
                for _ in range(self.options["model_updates"]):
                    self.parameters, metrics = self._model_update(
                        self.parameters, self.sample(self.options["window_steps"])
                    )
            starts = jax.tree.map(lambda x: x[:, 0], self.sample())
            if self.method == "supermimic":
                self.parameters, loss, rollout = self._analytic_update(
                    self.parameters, starts
                )
                metrics = {**metrics, "actor_loss": loss}
            else:
                rollout, _, _ = self._imagine(self.parameters, starts)
        if self.method == "visual_dagger":
            batch = jax.tree.map(lambda x: x[:, 0], self.sample())
            self.parameters, loss = self._dagger_update(self.parameters, batch)
            metrics["actor_loss"] = loss
            return metrics
        targets, advantage = self.critic_targets(
            self.parameters, rollout, simulator=self.method == "ppo_sim"
        )
        flat = jax.tree.map(
            lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])), rollout
        )
        targets, advantage = targets.reshape(-1), advantage.reshape(-1)
        if self.options["use_critic"]:
            # Targets are computed once and held fixed across all critic epochs.
            for _ in range(self.options["critic_epochs"]):
                self.parameters, loss = self._critic_update(
                    self.parameters, flat, targets
                )
            metrics["critic_loss"] = loss
        if self.method in {"ppo_sim", "ppo_wm"}:
            advantage = (advantage - jp.mean(advantage)) / (jp.std(advantage) + 1e-6)
            weights = jp.ones(rollout["reward"].shape)
            if self.method == "ppo_wm":
                weights = jp.concatenate(
                    (
                        jp.ones_like(weights[:1]),
                        jp.cumprod(1 - rollout["terminal"][:-1], axis=0),
                    ),
                    axis=0,
                )
            for _ in range(self.options["ppo_epochs"]):
                self.parameters, loss = self._ppo_update(
                    self.parameters, flat, advantage, weights.reshape(-1)
                )
            metrics["actor_loss"] = loss
        self.parameters = self.parameters.replace(
            rng=jax.random.split(self.parameters.rng)[0]
        )
        return metrics

    def diagnostics(self):
        result = {}
        for horizon in sorted(
            {1, self.options["window_steps"], self.options["horizon_steps"]}
        ):
            try:
                batch = self.sample(horizon, holdout=True)
            except ValueError as error:
                result[str(horizon)] = {"available": False, "reason": str(error)}
                continue
            _, losses = self.model_loss(self.parameters.model, batch)
            state = batch["state"][:, 0]
            for t in range(horizon):
                state, _, _ = self.predict(
                    self.parameters.model,
                    state,
                    batch["action"][:, t],
                    batch["goal"][:, t],
                    batch["context"][:, t],
                )
            final_rmse = jp.sqrt(
                jp.mean(((state - batch["next_state"][:, -1]) / self.scale) ** 2)
            )
            result[str(horizon)] = dict(
                available=True,
                final_state_rmse=float(final_rmse),
                mean_state_mse=float(losses[0]),
                reward_mse=float(losses[1]),
                terminal_bce=float(losses[2]),
                seconds=horizon * self.task.contract.control_dt,
            )
            per_window_mse = jp.mean(
                ((state - batch["next_state"][:, -1]) / self.scale) ** 2, axis=-1
            )
            for label in ("contact_change", "slip_proxy", "dropped"):
                selected = jp.any(batch[label].astype(bool), axis=1)
                count = int(jp.sum(selected))
                result[str(horizon)][label] = dict(
                    windows=count,
                    final_state_rmse=float(
                        jp.sqrt(jp.sum(jp.where(selected, per_window_mse, 0)) / count)
                    )
                    if count
                    else None,
                )
        if self.options["diagnostics"]["gradient_probe"]:
            from learning.supermimic.diagnostics import gradient_probe

            result["gradient_probe"] = gradient_probe(self)
        return result

    def evaluate(self):
        options = self.options["evaluation"]
        results = []
        self.last_evaluation_simulator_transitions = 0
        for first in range(0, options["episodes"], self.n):
            seeds = np.arange(
                options["seed_start"] + first, options["seed_start"] + first + self.n
            )
            keys = jax.vmap(lambda seed: jax.random.split(jax.random.PRNGKey(seed))[0])(
                jp.asarray(seeds)
            )
            environment = self.task.reset(keys)
            previous = jp.zeros_like(self.previous)
            active = np.ones(self.n, dtype=bool)
            returns = np.zeros(self.n)
            lengths = np.zeros(self.n, dtype=int)
            successes = np.zeros(self.n, dtype=bool)
            drops = np.zeros(self.n)
            errors = np.zeros(self.n)
            time_to_success = np.full(self.n, np.nan)
            for _ in range(self.config["task"]["env"]["episode_length"]):
                state, goal, context = self.describe(environment)
                pixels = self.render_batch(environment) if self.visual else None
                action = self._mean_action(
                    self.parameters.actor,
                    state,
                    goal,
                    context,
                    previous,
                    jp.asarray(pixels) if self.visual else None,
                )
                following = self.task.step(environment, action)
                self.last_evaluation_simulator_transitions += self.n
                environment = select_reset(environment, following, jp.asarray(active))
                executed = self.task.executed(environment)
                done = np.asarray(environment.done).astype(bool)
                previous = jp.where(jp.asarray(done)[:, None], 0.0, executed)
                returns += np.where(active, np.asarray(environment.reward), 0.0)
                lengths += active
                success = np.asarray(environment.metrics["is_success"]).astype(bool)
                first_success = active & success & ~successes
                time_to_success[first_success] = (
                    lengths[first_success] * self.task.contract.control_dt
                )
                successes |= active & success
                drops = np.maximum(
                    drops,
                    np.asarray(environment.metrics.get("dropped", np.zeros(self.n)))
                    * active,
                )
                errors += (
                    np.asarray(
                        environment.metrics.get(
                            "orientation_or_velocity_error", np.zeros(self.n)
                        )
                    )
                    * active
                )
                active &= ~done
                if not active.any():
                    break
            for i, seed in enumerate(seeds):
                if first + i >= options["episodes"]:
                    break
                metrics = dict(
                    dropped=float(drops[i]),
                    mean_orientation_or_velocity_error=float(
                        errors[i] / max(1, lengths[i])
                    ),
                )
                if np.isfinite(time_to_success[i]):
                    metrics["time_to_success_seconds"] = float(time_to_success[i])
                results.append(
                    EpisodeResult(
                        int(seed),
                        int(lengths[i]),
                        float(returns[i]),
                        True,
                        False,
                        bool(successes[i]),
                        metrics,
                    )
                )
        return results

    def save(self, directory):
        from learning.supermimic.policy import export_policy

        return export_policy(self, directory)


def train(config, output_dir):
    validate_config(config)
    started = time.monotonic()
    task = Task(config)
    try:
        trainer = Trainer(config, task)
        run_training(trainer, Path(output_dir), started=started)
    finally:
        task.close()


def run_training(trainer, output_dir, *, started=None):
    options = trainer.options
    started = time.monotonic() if started is None else started
    setup_seconds = time.monotonic() - started
    evaluation_seconds = 0.0
    warmed_iterations = []
    warmup = max(options["window_steps"], options["warmup_steps"])
    warm_started = time.monotonic()
    trainer.collect(warmup, 0)
    compile_and_warmup = time.monotonic() - warm_started
    first_iteration_seconds = None
    target_time = None
    log_path = output_dir / "metrics.jsonl"
    imagined = 0
    model_fit_transitions = 0
    evaluation_transitions = 0
    with log_path.open("x") as stream:
        for iteration in range(1, options["iterations"] + 1):
            tick = time.monotonic()
            metrics = trainer.train_iteration(iteration)
            metrics = {k: float(v) for k, v in metrics.items()}
            iteration_seconds = time.monotonic() - tick
            if iteration == 1:
                first_iteration_seconds = iteration_seconds
            else:
                warmed_iterations.append(iteration_seconds)
            if trainer.method in {"supermimic", "ppo_wm"}:
                imagined += options["batch_size"] * options["horizon_steps"]
                if not options["freeze_model"]:
                    model_fit_transitions += (
                        options["model_updates"]
                        * options["batch_size"]
                        * options["window_steps"]
                    )
            record = dict(
                iteration=iteration,
                **metrics,
                simulator_transitions=trainer.replay.steps * trainer.n,
                training_transitions=trainer.replay.steps * trainer.train_n,
                imagined_transitions=imagined,
                model_fit_transitions=model_fit_transitions,
                elapsed_seconds=time.monotonic() - started,
                iteration_training_seconds=iteration_seconds,
            )
            if (
                iteration % options["evaluation_interval"] == 0
                or iteration == options["iterations"]
            ):
                eval_started = time.monotonic()
                results = trainer.evaluate()
                evaluation_transitions += trainer.last_evaluation_simulator_transitions
                evaluation_dir = output_dir / f"evaluation_{iteration:07d}"
                write_results(results, evaluation_dir)
                record["evaluation_seconds"] = time.monotonic() - eval_started
                record["success_rate"] = float(np.mean([r.success for r in results]))
                record["simulator_return"] = float(
                    np.mean([r.total_reward for r in results])
                )
                if trainer.method in {"supermimic", "ppo_wm"}:
                    record["heldout_model"] = trainer.diagnostics()
                if (
                    record["success_rate"]
                    >= options["evaluation"]["target_success_rate"]
                    and target_time is None
                ):
                    target_time = time.monotonic() - started
                evaluation_seconds += time.monotonic() - eval_started
            if iteration % options["checkpoint_interval"] == 0:
                trainer.save(output_dir / f"checkpoint_{iteration:07d}")
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
    trainer.save(output_dir / "final")
    summary = dict(
        method=trainer.method,
        observation=options["observation"],
        iterations=options["iterations"],
        simulator_transitions=trainer.replay.steps * trainer.n,
        imagined_transitions=imagined,
        model_fit_transitions=model_fit_transitions,
        evaluation_simulator_transitions=evaluation_transitions,
        elapsed_seconds=time.monotonic() - started,
        setup_seconds=setup_seconds,
        evaluation_and_diagnostics_seconds=evaluation_seconds,
        training_seconds=time.monotonic() - started - evaluation_seconds,
        explicit_kernel_compile_seconds=sum(
            value.compile_seconds
            for value in vars(trainer).values()
            if isinstance(value, CompiledKernel)
        ),
        warmed_iteration_seconds_mean=float(np.mean(warmed_iterations))
        if warmed_iterations
        else None,
        collection_compile_and_warmup_seconds=compile_and_warmup,
        first_iteration_including_compilation_seconds=first_iteration_seconds,
        time_to_target_seconds=target_time,
        control_dt=trainer.task.contract.control_dt,
        window_seconds=options["window_steps"] * trainer.task.contract.control_dt,
        imagination_seconds=options["horizon_steps"] * trainer.task.contract.control_dt,
        evaluation_policy="learner_only_in_simulator",
        convergence_claim=False,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
