"""Differentiable survival returns, fixed lambda targets and squashed PPO."""

import jax
import jax.numpy as jp

LOG_2PI = jp.log(2 * jp.pi)


def action_log_prob(mean, log_std, raw):
    normal = -0.5 * (((raw - mean) / jp.exp(log_std)) ** 2 + 2 * log_std + LOG_2PI)
    # Stable log(1-tanh(raw)^2).
    correction = 2 * (jp.log(2.0) - raw - jax.nn.softplus(-2 * raw))
    return jp.sum(normal - correction, axis=-1)


def lambda_returns(rewards, terminals, values, gamma, gae_lambda):
    """Time-major arrays; values includes the final bootstrap state.

    Terminals are physical termination labels/probabilities, not time limits.
    The caller stops gradients on targets during critic fitting.
    """

    def body(carry, inputs):
        reward, terminal, value = inputs
        target = reward + gamma * (1 - terminal) * (
            (1 - gae_lambda) * value + gae_lambda * carry
        )
        return target, target

    _, targets = jax.lax.scan(
        body, values[-1], (rewards, terminals, values[1:]), reverse=True
    )
    return targets


def survival_return(rewards, terminals, final_value, gamma):
    """Reward on the terminating transition counts; subsequent rewards do not."""

    def body(carry, inputs):
        survival, discount, total = carry
        reward, terminal = inputs
        return (
            survival * (1 - terminal),
            discount * gamma,
            total + discount * survival * reward,
        ), None

    (survival, discount, total), _ = jax.lax.scan(
        body,
        (
            jp.ones_like(final_value),
            jp.ones_like(final_value),
            jp.zeros_like(final_value),
        ),
        (rewards, terminals),
    )
    return total + survival * discount * final_value


def ppo_loss(log_probability, old_log_probability, advantage, clip_ratio):
    ratio = jp.exp(log_probability - old_log_probability)
    return -jp.mean(
        jp.minimum(
            ratio * advantage,
            jp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
        )
    )
