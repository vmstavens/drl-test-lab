"""Small configurable networks; learned pixels are an explicit extension."""

import jax.numpy as jp
from flax import linen as nn


class MLP(nn.Module):
    widths: tuple[int, ...]
    output_size: int

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.elu(nn.Dense(width)(x))
        return nn.Dense(self.output_size)(x)


class Actor(nn.Module):
    action_size: int
    widths: tuple[int, ...]
    visual: bool = False
    initial_std: float = 0.1

    @nn.compact
    def __call__(self, features, pixels=None):
        if self.visual:
            x = pixels
            for width in (16, 32, 32):
                x = nn.elu(nn.Conv(width, (3, 3), strides=(2, 2))(x))
            x = x.reshape((*x.shape[:-3], -1))
            features = jp.concatenate((features, x), axis=-1)
        mean = MLP(self.widths, self.action_size)(features)
        log_std = self.param(
            "log_std",
            lambda key, shape: jp.full(shape, jp.log(self.initial_std)),
            (self.action_size,),
        )
        return mean, jp.clip(log_std, -5.0, 0.0)


class WorldModel(nn.Module):
    state_size: int
    widths: tuple[int, ...]

    def setup(self):
        self.dynamics = MLP(self.widths, self.state_size)
        self.heads = MLP(self.widths[-1:], 2)

    def __call__(self, state, action, context):
        return state + self.dynamics(jp.concatenate((state, action, context), axis=-1))

    def outcomes(self, state, action, following, goal, context):
        heads = self.heads(
            jp.concatenate((state, action, following, goal, context), axis=-1)
        )
        return heads[..., 0], heads[..., 1]

    def initialize(self, state, action, context, goal):
        following = self(state, action, context)
        reward, terminal = self.outcomes(state, action, following, goal, context)
        return following, reward, terminal


class PixelModel(nn.Module):
    shape: tuple[int, int, int]
    widths: tuple[int, ...]

    @nn.compact
    def __call__(self, state, context):
        output = MLP(self.widths, self.shape[0] * self.shape[1] * 3)(
            jp.concatenate((state, context), axis=-1)
        )
        return nn.sigmoid(output.reshape((*output.shape[:-1], *self.shape)))
