"""Compare learned-model improvement directions against the physical simulator."""

import jax
import jax.numpy as jp
import numpy as np

from learning.supermimic.objectives import survival_return


def gradient_probe(trainer):
    """A short open-loop directional check, using fresh diagnostic reset keys.

    Contact makes finite differences nonsmooth. This is a diagnostic, not a gate
    asserting that a learned dynamics Jacobian equals a simulator derivative.
    """
    options = trainer.options["diagnostics"]
    horizon = options["probe_steps"]
    seed = trainer.options["evaluation"]["seed_start"]
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed + i), 2))(
        jp.arange(trainer.n)
    )
    environment = trainer.task.reset(keys)
    initial, goal, context = trainer.describe(environment)
    previous = jp.zeros_like(trainer.previous)
    pixels = trainer.render_batch(environment) if trainer.visual else None
    command = trainer._mean_action(
        trainer.parameters.actor,
        initial,
        goal,
        context,
        previous,
        jp.asarray(pixels) if trainer.visual else None,
    )
    commands = jp.broadcast_to(command, (horizon, *command.shape))

    def model_objective(commands):
        def body(state, requested):
            executed = trainer.task.model_action(state, requested)
            following, reward, logit = trainer.predict(
                trainer.parameters.model, state, executed, goal, context
            )
            return following, (reward, jax.nn.sigmoid(logit))

        _, (rewards, terminals) = jax.lax.scan(body, initial, commands)
        return jp.mean(
            survival_return(
                rewards, terminals, jp.zeros(trainer.n), trainer.options["gamma"]
            )
        )

    @jax.jit
    def simulator_objective(commands):
        def body(carry, requested):
            state, alive = carry
            following = trainer.task.step(state, requested)
            reward = following.reward / trainer.reward_scale * alive
            alive = alive * (1 - following.done)
            return (following, alive), reward

        _, rewards = jax.lax.scan(body, (environment, jp.ones(trainer.n)), commands)
        discount = trainer.options["gamma"] ** jp.arange(horizon)
        return jp.mean(jp.sum(rewards * discount[:, None], axis=0))

    model_value, gradient = jax.jit(jax.value_and_grad(model_objective))(commands)
    norm = jp.linalg.norm(gradient)
    direction = gradient / jp.maximum(norm, 1e-8)
    epsilon = options["probe_epsilon"]
    plus, minus = (
        jp.clip(commands + epsilon * direction, -1, 1),
        jp.clip(commands - epsilon * direction, -1, 1),
    )
    actual = simulator_objective(commands)
    simulator_slope = (simulator_objective(plus) - simulator_objective(minus)) / (
        2 * epsilon
    )
    model_slope = (model_objective(plus) - model_objective(minus)) / (2 * epsilon)
    values = dict(
        steps=horizon,
        seconds=horizon * trainer.task.contract.control_dt,
        model_return=float(model_value * trainer.reward_scale),
        simulator_return=float(actual * trainer.reward_scale),
        return_gap=float((model_value - actual) * trainer.reward_scale),
        model_gradient_norm=float(norm),
        model_directional_slope=float(model_slope),
        simulator_directional_slope=float(simulator_slope),
        direction_improves_simulator=bool(simulator_slope > 0),
    )
    if not all(np.isfinite(v) for v in values.values()):
        raise FloatingPointError("Non-finite learned-model gradient probe")
    return values
