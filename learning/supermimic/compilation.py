"""Account for explicit kernel compilation separately from execution."""

import time

import jax


class CompiledKernel:
    def __init__(self, function):
        self.function = jax.jit(function)
        self.cache = {}
        self.compile_seconds = 0.0

    def __call__(self, *args):
        leaves, tree = jax.tree.flatten(args)
        # PyTreeDef equality handles MuJoCo's immutable array metadata. Its string
        # representation can contain temporary object addresses and is not a key.
        signature = (tree, tuple((tuple(x.shape), str(x.dtype)) for x in leaves))
        if signature not in self.cache:
            started = time.monotonic()
            self.cache[signature] = self.function.lower(*args).compile()
            self.compile_seconds += time.monotonic() - started
        return self.cache[signature](*args)
