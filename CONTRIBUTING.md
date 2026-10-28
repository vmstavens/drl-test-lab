# Contributing

Use Python 3.11 and `uv sync --frozen --inexact`. Select JAX CUDA support
manually; preserve extra GPU packages with `uv run --no-sync`.

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
```

The CPU suite exercises the actual compiled learner on the small MuJoCo task,
including fused/split update equivalence, critic-only warmup, executed-action
history, reference selection, evaluation isolation and deployment contracts.
GPU throughput measurements are separate from correctness and convergence.

Keep workflows thin and route execution through `execute_workflow`. Document
new experiment choices and compatibility changes. Keep tasks independent of
algorithms and preserve tests of the semantic contract. Generated data, weights,
run outputs and credentials remain outside Git.
