"""One compiled DRLR/SAC path with executed history and a frozen reference.

The core owns no task-specific frames or legacy checkpoint conversions. A
reference is a pure callable over Task's explicit observable/controller state.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import flax.linen as nn
import flax.serialization
import flax.struct
import jax
import jax.numpy as jp
import numpy as np
import optax

from learning.reinforcement.policy_rollouts import make_policy_rollout_runner


class Actor(nn.Module):
    observable_dim: int
    action_dim: int
    hidden_sizes: tuple
    encoder_sizes: tuple
    latent_dim: int
    initial_std: float
    minimum_std: float
    maximum_std: float

    def setup(self):
        self.encoder = [
            nn.Dense(width, name=f"encoder_{i}")
            for i, width in enumerate((*self.encoder_sizes, self.latent_dim))
        ]
        self.layers = [
            nn.Dense(
                width,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                name=f"actor_{i}",
            )
            for i, width in enumerate(self.hidden_sizes)
        ]
        self.mean = nn.Dense(
            self.action_dim,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="mean",
        )
        fraction = (self.initial_std - self.minimum_std) / (
            self.maximum_std - self.minimum_std
        )
        self.scale = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(np.log(fraction / (1 - fraction))),
            name="scale",
        )

    def encode(self, factors):
        x = factors
        for layer in self.encoder[:-1]:
            x = nn.elu(layer(x))
        return self.encoder[-1](x)

    def __call__(self, observation, latent=None):
        latent = (
            self.encode(observation[..., self.observable_dim :])
            if latent is None
            else latent
        )
        x = jp.concatenate((observation[..., : self.observable_dim], latent), axis=-1)
        for layer in self.layers:
            x = nn.elu(layer(x))
        mean = self.mean(x)
        std = self.minimum_std + (self.maximum_std - self.minimum_std) * jax.nn.sigmoid(
            self.scale(x)
        )
        return mean, std


class TwinQ(nn.Module):
    hidden_sizes: tuple
    layer_norm: bool

    @nn.compact
    def __call__(self, observation, action):
        inputs = jp.concatenate((observation, action), axis=-1)
        values = []
        for critic in range(2):
            x = inputs
            for i, width in enumerate(self.hidden_sizes):
                x = nn.Dense(width, name=f"q{critic}_dense{i}")(x)
                if self.layer_norm:
                    x = nn.LayerNorm(name=f"q{critic}_norm{i}")(x)
                x = nn.elu(x)
            values.append(nn.Dense(1, name=f"q{critic}_out")(x)[..., 0])
        return jp.stack(values, axis=-1)


def sample_action(actor, params, observation, key):
    mean, std = actor.apply(params, observation)
    noise = jax.random.normal(key, mean.shape)
    pre = mean + std * noise
    action = jp.tanh(pre)
    log_prob = -0.5 * noise**2 - jp.log(std) - 0.5 * np.log(2 * np.pi)
    correction = 2 * (np.log(2) - pre - jax.nn.softplus(-2 * pre))
    return action, jp.sum(log_prob - correction, axis=-1)


def prefer_reference(learner_q, reference_q):
    """The learner must strictly beat the reference on both target critics."""
    return jp.any(learner_q <= reference_q, axis=-1, keepdims=True)


@flax.struct.dataclass
class TrainState:
    policy_params: object
    q_params: object
    target_q_params: object
    actor_optimizer: object
    critic_optimizer: object
    log_alpha: object
    alpha_optimizer: object
    environment: object
    previous_action: object
    replay: object
    replay_position: object
    replay_size: object
    rng: object
    vector_steps: object
    updates: object


def empty_replay(capacity, width, action_size):
    return dict(
        observation=jp.zeros((capacity, width)),
        action=jp.zeros((capacity, action_size)),
        reward=jp.zeros(capacity),
        discount=jp.zeros(capacity),
        next_observation=jp.zeros((capacity, width)),
    )


def summarize_metrics(metrics):
    result = jax.tree.map(jp.mean, metrics)
    for name in ("completed", "successes"):
        result[name] = jp.sum(metrics[name])
    return result


class Trainer:
    def __init__(self, config, task, reference, *, center, scale):
        self.config, self.task, self.reference = config, task, reference
        self.rl = config["rl"]
        self.center, self.scale = (
            jp.asarray(center),
            jp.maximum(jp.asarray(scale), 0.001),
        )
        self.o, self.a, self.f = (
            task.observation_size,
            task.action_size,
            self.rl["privileged_dim"],
        )
        self.width = self.o + self.a + self.f
        if self.f != len(task.contract.privileged_layout):
            raise ValueError("Privileged factor layout differs from the learner")
        if (
            self.rl["batch_size"] < 2
            or self.rl["gradient_steps"] < 1
            or self.rl["memory_size"] < self.rl["num_envs"]
        ):
            raise ValueError("Batch, updates and replay capacity must support training")
        if self.rl["warmup_vector_steps"] < 0:
            raise ValueError("Warmup cannot be negative")
        gate = self.rl["evaluation_gate"]
        if (
            not 0 < gate["success_threshold"] <= 1
            or gate["episodes"] < 1
            or gate["progress_episodes"] < 1
        ):
            raise ValueError("Evaluation counts and success threshold must be positive")
        promotion = range(gate["seed_start"], gate["seed_start"] + gate["episodes"])
        progress = range(
            gate["progress_seed_start"],
            gate["progress_seed_start"] + gate["progress_episodes"],
        )
        for name in ("bootstrap", "handoff"):
            spec = self.rl[name]
            seeds = range(spec["seed_start"], spec["seed_start"] + spec["episodes"])
            if max(seeds.start, promotion.start) < min(
                seeds.stop, promotion.stop
            ) or max(seeds.start, progress.start) < min(seeds.stop, progress.stop):
                raise ValueError("Training reference seeds overlap evaluation seeds")
        if max(promotion.start, progress.start) < min(promotion.stop, progress.stop):
            raise ValueError("Progress and promotion evaluation seeds overlap")
        if self.rl["previous_action_source"] != "executed":
            raise ValueError("Only executed-action history is supported")
        if self.rl["training_execution"] not in {"fused", "split"}:
            raise ValueError("training_execution must be fused or split")
        if not 0 < self.rl["reference_replay_fraction"] < 1:
            raise ValueError("Successful reference replay fraction must be in (0,1)")
        self.actor = Actor(
            self.o + self.a,
            self.a,
            tuple(self.rl["actor_hidden_sizes"]),
            tuple(self.rl["encoder_hidden_sizes"]),
            self.rl["latent_dim"],
            self.rl["initial_action_std"],
            self.rl["minimum_action_std"],
            self.rl["maximum_action_std"],
        )
        self.critic = TwinQ(
            tuple(self.rl["critic_hidden_sizes"]), self.rl["critic_layer_norm"]
        )
        self.actor_optimizer = optax.adam(self.rl["actor_learning_rate"])
        self.critic_optimizer = optax.adam(self.rl["critic_learning_rate"])
        self.alpha_optimizer = optax.adam(self.rl["entropy_learning_rate"])
        self._step = jax.jit(self.environment_step)
        self._update = jax.jit(self.gradient_update)
        self._chunks = {}

    def features(self, observation, previous):
        return jp.concatenate(
            (
                (observation[..., : self.o] - self.center) / self.scale,
                previous,
                observation[..., self.o : self.o + self.f],
            ),
            axis=-1,
        )

    def teacher(self, features, previous_observation, key):
        raw = features[..., : self.o] * self.scale + self.center
        return self.reference(
            raw,
            previous_observation,
            features[..., self.o : self.o + self.a],
            features[..., self.o + self.a :],
            key,
        )

    def initialize(self, *, parent=None):
        key, pkey, qkey, reset_key = jax.random.split(
            jax.random.PRNGKey(self.config["seed"]), 4
        )
        env = self.task.reset(jax.random.split(reset_key, self.rl["num_envs"]))
        if env.obs.shape[-1] != self.o * 2 + self.f:
            raise ValueError(
                "Task observable/privileged layout differs from configuration"
            )
        dummy = jp.zeros((1, self.width))
        params = self.actor.init(pkey, dummy) if parent is None else parent
        q = self.critic.init(qkey, dummy, jp.zeros((1, self.a)))
        log_alpha = jp.array(
            np.log(self.rl["initial_entropy_temperature"]), dtype=jp.float32
        )
        return TrainState(
            params,
            q,
            q,
            self.actor_optimizer.init(params),
            self.critic_optimizer.init(q),
            log_alpha,
            self.alpha_optimizer.init(log_alpha),
            env,
            jp.zeros((self.rl["num_envs"], self.a)),
            empty_replay(self.rl["memory_size"], self.width, self.a),
            jp.array(0),
            jp.array(0),
            key,
            jp.array(0),
            jp.array(0),
        )

    def environment_step(self, state):
        key, action_key, teacher_key, reset_key = jax.random.split(state.rng, 4)
        obs = self.features(state.environment.obs, state.previous_action)
        candidate, _ = sample_action(self.actor, state.policy_params, obs, action_key)
        reference = self.teacher(
            obs, state.environment.obs[..., -self.o :], teacher_key
        )
        use_reference = prefer_reference(
            self.critic.apply(state.target_q_params, obs, candidate),
            self.critic.apply(state.target_q_params, obs, reference),
        )
        use_reference = use_reference | (
            state.vector_steps < self.rl["warmup_vector_steps"]
        )
        requested = jp.where(use_reference, reference, candidate)
        following = self.task.step(state.environment, requested)
        executed = self.task.executed(following)
        done = following.done.astype(bool)
        next_previous = jp.where(done[:, None], 0.0, executed)
        transition = dict(
            observation=obs,
            action=executed,
            reward=following.reward,
            discount=1.0 - following.done,
            next_observation=self.features(following.obs, next_previous),
        )
        indices = (state.replay_position + jp.arange(self.rl["num_envs"])) % self.rl[
            "memory_size"
        ]
        replay = jax.tree.map(
            lambda buffer, data: buffer.at[indices].set(data), state.replay, transition
        )
        fresh = self.task.reset(jax.random.split(reset_key, self.rl["num_envs"]))

        def reset_slot(old, new):
            if (
                hasattr(old, "ndim")
                and old.ndim
                and old.shape[0] == self.rl["num_envs"]
            ):
                return jp.where(
                    done.reshape((len(done),) + (1,) * (old.ndim - 1)), new, old
                )
            return old

        env = jax.tree.map(reset_slot, following, fresh)
        metrics = dict(
            completed=jp.sum(done),
            successes=jp.sum(done & (following.reward > 0)),
            teacher_fraction=jp.mean(use_reference.astype(jp.float32)),
            actor_loss=jp.array(0.0),
            critic_loss=jp.array(0.0),
            bellman_target_mean=jp.array(0.0),
        )
        return state.replace(
            environment=env,
            previous_action=next_previous,
            replay=replay,
            replay_position=(state.replay_position + self.rl["num_envs"])
            % self.rl["memory_size"],
            replay_size=jp.minimum(
                state.replay_size + self.rl["num_envs"], self.rl["memory_size"]
            ),
            rng=key,
            vector_steps=state.vector_steps + 1,
        ), metrics

    def gradient_update(self, state, reference_replay):
        key, online_key, reference_key, target_key, teacher_key, actor_key = (
            jax.random.split(state.rng, 6)
        )
        reference_count = round(
            self.rl["batch_size"] * self.rl["reference_replay_fraction"]
        )
        online_count = self.rl["batch_size"] - reference_count
        online_indices = jax.random.randint(
            online_key, (online_count,), 0, state.replay_size
        )
        reference_indices = jax.random.randint(
            reference_key, (reference_count,), 0, len(reference_replay["reward"])
        )
        batch = jax.tree.map(
            lambda online, ref: jp.concatenate(
                (online[online_indices], ref[reference_indices])
            ),
            state.replay,
            reference_replay,
        )
        alpha = jp.exp(state.log_alpha)
        candidate, log_prob = sample_action(
            self.actor, state.policy_params, batch["next_observation"], target_key
        )
        previous_raw = batch["observation"][..., : self.o] * self.scale + self.center
        teacher = self.teacher(batch["next_observation"], previous_raw, teacher_key)
        q_rl = self.critic.apply(
            state.target_q_params, batch["next_observation"], candidate
        )
        q_ref = self.critic.apply(
            state.target_q_params, batch["next_observation"], teacher
        )
        use_reference = prefer_reference(q_rl, q_ref) | (
            state.vector_steps <= self.rl["warmup_vector_steps"]
        )
        selected_q = self.critic.apply(
            state.target_q_params,
            batch["next_observation"],
            jp.where(use_reference, teacher, candidate),
        )
        value = jp.min(selected_q, axis=-1) - jp.where(
            use_reference[:, 0], 0.0, alpha * log_prob
        )
        target = jp.clip(
            batch["reward"] + self.rl["discount_factor"] * batch["discount"] * value,
            self.rl["bellman_target_min"],
            self.rl["bellman_target_max"],
        )
        target = jax.lax.stop_gradient(target)

        def q_loss(params):
            return 0.5 * jp.mean(
                (
                    self.critic.apply(params, batch["observation"], batch["action"])
                    - target[:, None]
                )
                ** 2
            )

        critic_loss, gradient = jax.value_and_grad(q_loss)(state.q_params)
        changes, q_opt = self.critic_optimizer.update(
            gradient, state.critic_optimizer, state.q_params
        )
        q_params = optax.apply_updates(state.q_params, changes)

        def actor_loss(params):
            action, probability = sample_action(
                self.actor, params, batch["observation"], actor_key
            )
            value = self.critic.apply(q_params, batch["observation"], action)
            return jp.mean(alpha * probability - jp.min(value, axis=-1)), probability

        def update_actor(_):
            (loss, probability), gradient = jax.value_and_grad(
                actor_loss, has_aux=True
            )(state.policy_params)
            changes, optimizer = self.actor_optimizer.update(
                gradient, state.actor_optimizer, state.policy_params
            )
            return (
                optax.apply_updates(state.policy_params, changes),
                optimizer,
                loss,
                probability,
            )

        def frozen_actor(_):
            _, probability = sample_action(
                self.actor, state.policy_params, batch["observation"], actor_key
            )
            return (
                state.policy_params,
                state.actor_optimizer,
                jp.array(0.0),
                probability,
            )

        params, actor_opt, loss, probability = jax.lax.cond(
            state.vector_steps <= self.rl["warmup_vector_steps"],
            frozen_actor,
            update_actor,
            None,
        )
        log_alpha, alpha_opt = state.log_alpha, state.alpha_optimizer
        if not self.rl["freeze_entropy_temperature"]:

            def temperature_loss(value):
                return jp.mean(
                    jp.exp(value) * jax.lax.stop_gradient(-probability + self.a)
                )

            def update_temperature(values):
                value, optimizer_state = values
                gradient = jax.grad(temperature_loss)(value)
                changes, optimizer_state = self.alpha_optimizer.update(
                    gradient, optimizer_state, value
                )
                return optax.apply_updates(value, changes), optimizer_state

            log_alpha, alpha_opt = jax.lax.cond(
                state.vector_steps <= self.rl["warmup_vector_steps"],
                lambda values: values,
                update_temperature,
                (log_alpha, alpha_opt),
            )
        target_q = optax.incremental_update(
            q_params, state.target_q_params, self.rl["polyak"]
        )
        return state.replace(
            policy_params=params,
            q_params=q_params,
            target_q_params=target_q,
            actor_optimizer=actor_opt,
            critic_optimizer=q_opt,
            log_alpha=log_alpha,
            alpha_optimizer=alpha_opt,
            rng=key,
            updates=state.updates + 1,
        ), dict(
            actor_loss=loss,
            critic_loss=critic_loss,
            bellman_target_mean=jp.mean(target),
        )

    def train_chunk(self, state, reference, steps):
        if self.rl["training_execution"] == "split":
            metrics = []
            for _ in range(steps):
                state, metric = self._step(state)
                count = (
                    1
                    if int(state.vector_steps) <= self.rl["warmup_vector_steps"]
                    else self.rl["gradient_steps"]
                )
                update_metrics = []
                for _ in range(count):
                    state, losses = self._update(state, reference)
                    update_metrics.append(losses)
                losses = jax.tree.map(
                    lambda *values: jp.mean(jp.stack(values)), *update_metrics
                )
                metrics.append({**metric, **losses})
            stacked = jax.tree.map(lambda *values: jp.stack(values), *metrics)
            return state, summarize_metrics(stacked)
        if steps not in self._chunks:

            def chunk(state, reference):
                def body(state, _):
                    state, metrics = self.environment_step(state)

                    def update(carry, i):
                        def skip(s):
                            return s, dict(
                                actor_loss=jp.array(0.0),
                                critic_loss=jp.array(0.0),
                                bellman_target_mean=jp.array(0.0),
                            )

                        return jax.lax.cond(
                            (state.vector_steps <= self.rl["warmup_vector_steps"])
                            & (i > 0),
                            skip,
                            lambda s: self.gradient_update(s, reference),
                            carry,
                        )

                    state, losses = jax.lax.scan(
                        update, state, jp.arange(self.rl["gradient_steps"])
                    )
                    return state, {
                        **metrics,
                        **jax.tree.map(
                            lambda value: (
                                jp.sum(value)
                                / jp.where(
                                    state.vector_steps
                                    <= self.rl["warmup_vector_steps"],
                                    1,
                                    self.rl["gradient_steps"],
                                )
                            ),
                            losses,
                        ),
                    }

                state, metrics = jax.lax.scan(body, state, None, length=steps)
                return state, summarize_metrics(metrics)

            self._chunks[steps] = jax.jit(chunk)
        return self._chunks[steps](state, reference)

    def rollout(self, params, *, episodes, seed_start, capture=False):
        def action(norm, params, obs, previous, valid, key):
            features = self.features(obs, previous)
            mean, _ = self.actor.apply(params, features)
            return jp.tanh(mean)

        run = make_policy_rollout_runner(
            self.task,
            jax.jit(action),
            self.features,
            episodes=episodes,
            batch_size=self.rl["evaluation_gate"]["batch_size"],
            episode_length=self.config["task"]["env"]["episode_length"],
            action_size=self.a,
            capture_successes=capture,
            execution="compiled",
            chunk_steps=self.rl["chunk_steps"],
        )
        summary, paths = run(
            SimpleNamespace(normalizer_state=(), policy_params=params),
            seed_start=seed_start,
        )
        return {**summary, "evaluation_policy": "learner_only"}, paths

    def save_actor(self, path, state):
        path = Path(path)
        path.write_bytes(flax.serialization.to_bytes(state.policy_params))
        path.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    contract=self.task.contract.to_dict(),
                    config=self.config,
                    center=np.asarray(self.center).tolist(),
                    scale=np.asarray(self.scale).tolist(),
                ),
                indent=2,
            )
        )


def replay_from_paths(paths):
    if not paths:
        raise ValueError("Successful reference replay cannot be empty")
    result = {
        key: []
        for key in ("observation", "action", "reward", "discount", "next_observation")
    }
    for length, observation, action, following, reward in paths:
        result["observation"].append(observation)
        result["action"].append(action)
        result["reward"].append(reward)
        result["discount"].append(np.r_[np.ones(length - 1), 0.0].astype(np.float32))
        result["next_observation"].append(following)
    return {key: jp.asarray(np.concatenate(values)) for key, values in result.items()}


def train_drlr2(config, output_dir):
    from learning.reinforcement.recursive import train_recursive

    return train_recursive(config, output_dir)
