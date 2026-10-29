"""Bounded replay of contiguous, episode-safe windows, split by environment."""

import numpy as np


class SequenceReplay:
    def __init__(self, capacity_steps, num_envs, holdout_envs):
        if capacity_steps < 1 or not 0 < holdout_envs < num_envs:
            raise ValueError(
                "Replay requires capacity and disjoint training/holdout lanes"
            )
        self.capacity = capacity_steps
        self.num_envs = num_envs
        self.holdout_envs = holdout_envs
        self.steps = 0
        self.arrays = {}
        self.timestamps = np.full(capacity_steps, -1, dtype=np.int64)

    def add(self, transition):
        transition = {k: np.asarray(v) for k, v in transition.items()}
        if any(v.shape[0] != self.num_envs for v in transition.values()):
            raise ValueError("Every replay field must have one row per environment")
        if not self.arrays:
            self.arrays = {
                k: np.empty((self.capacity, *v.shape), dtype=v.dtype)
                for k, v in transition.items()
            }
        if set(transition) != set(self.arrays):
            raise ValueError("Replay field layout changed")
        slot = self.steps % self.capacity
        for k, value in transition.items():
            if value.shape != self.arrays[k].shape[1:]:
                raise ValueError(f"Replay shape changed: {k}")
            self.arrays[k][slot] = value
        self.timestamps[slot] = self.steps
        self.steps += 1

    def sample(self, rng, batch_size, length, *, holdout=False):
        if length < 1 or length > self.capacity or batch_size < 1:
            raise ValueError("Invalid replay window or batch size")
        oldest = max(0, self.steps - self.capacity)
        starts = np.arange(oldest, self.steps - length + 1)
        lanes = np.arange(
            self.num_envs - self.holdout_envs if holdout else 0,
            self.num_envs if holdout else self.num_envs - self.holdout_envs,
        )
        if not len(starts):
            raise ValueError("Not enough replay for a contiguous window")
        times = (starts[:, None] + np.arange(length)) % self.capacity
        valid = np.ones((len(starts), len(lanes)), dtype=bool)
        # A terminal transition may end a window, but may never be crossed.
        for offset in range(length - 1):
            valid &= ~self.arrays["done"][times[:, offset]][:, lanes].astype(bool)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            raise ValueError("Replay has no episode-safe windows of this length")
        chosen = rng.integers(0, len(rows), size=batch_size)
        indices = times[rows[chosen]]
        envs = lanes[cols[chosen]][:, None]
        return {k: v[indices, envs] for k, v in self.arrays.items()}
