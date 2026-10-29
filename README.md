# Research workbench

A runnable MuJoCo/MJX research template with demonstration collection, temporal
Diffusion Policy training, recursive DRLR with successful-reference replay, and
separate RMA history adaptation. The small `WorkbenchReach` task exercises the
complete pipeline without external assets or an existing dataset. The DLO
insertion environment remains available as an additional simulation example.

The learning recipe comes from the saved successful upstream insertion run,
with explicit task and implementation differences documented in
[the canonical learning setup](docs/canonical-learning-setup.md).
Implementation checks are recorded in [the validation report](docs/template-validation.md).
Passing implementation tests does **not** establish learning convergence or
reproduce the upstream success rates on this new task.

## Setup

Use Python 3.11 and the committed lockfile:

```bash
uv sync --frozen --inexact
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
```

JAX GPU support is selected manually for the installed driver. The default
lock selects base JAX; install the matching JAX CUDA extra explicitly if needed.
Use `uv sync --frozen --inexact` to preserve manually installed GPU packages and
`uv run --no-sync` for subsequent commands. Learning workflows default to CPU;
set `runtime.device=cuda` to select the GPU. No dependency on a sibling checkout
or the repository that supplied this implementation remains.

## Run the worked example

Every workflow defaults to dry-run. Preview the sole phase-one recipe:

```bash
uv run --no-sync python -m workflows.train_recursive_drlr_rma
```

Collect successful scripted demonstrations, prepare immutable transitions and
train DP. These commands use fresh default dataset paths; an existing dataset
is never silently overwritten:

```bash
uv run --no-sync python -m workflows.collect_demos runtime.dry_run=false
uv run --no-sync python -m workflows.prepare_dataset runtime.dry_run=false
uv run --no-sync python -m workflows.train_dp \
  runtime.dry_run=false hydra.run.dir=runs/dp
```

Start recursive phase-one training from the resulting DP:

```bash
uv run --no-sync python -m workflows.train_recursive_drlr_rma \
  runtime.dry_run=false il.checkpoint=runs/dp/models/final_model.pkl \
  hydra.run.dir=runs/recursive
```

The launcher first requires 40 successful DP rollouts and at least 80% success
in 64 attempts. Only learner-only evaluation can accept an actor. An accepted
actor becomes the initializer and frozen reference of the next domain level;
successful replay is retained. A failed gate stops recursion.

History adaptation is a separate workflow. Point it at an accepted stage:

```bash
uv run --no-sync python -m workflows.train_recursive_rma_phase2 \
  runtime.dry_run=false phase1_job=runs/recursive/stage_020 \
  hydra.run.dir=runs/history
```

Only the history encoder learns in phase two. Its final evaluation runs the
student alone. A bundle that fails that evaluation cannot be loaded for
deployment. `workflows.train_drlr2` is an alias for the canonical recipe, not a
second learning preset.

## Inspect, evaluate and record

```bash
uv run --no-sync python -m workflows.view_env
uv run --no-sync python -m tools.inspect_checkpoint runs/dp/policy.json
uv run --no-sync python -m workflows.evaluate runtime.dry_run=false \
  evaluation.policy_manifest=runs/history/policy.json
MUJOCO_GL=egl uv run --no-sync python -m workflows.render_rollout \
  runtime.dry_run=false evaluation.policy_manifest=runs/history/policy.json
```

The viewer/recorder and evaluation share the same episode runner. DP and RMA
exports include manifests with frames, action bounds and units, timing,
preprocessing and executed-history semantics. Equal dimensions alone do not
establish compatibility.

## Research infrastructure

- Explicit transition/vector-step budgets and control/physics timing.
- Episode-level dataset splits, training-only normalization and file hashes.
- Executed-action history, including reference-selected and clipped commands.
- Compiled rolling evaluation with immediate slot refills.
- Atomic recovery of simulation, replay, RNG, networks and optimizer state.
- TensorBoard, optional ntfy notifications, experiment matrices, statistical
  analysis, publication figures and reproduction bundles.

See [the tools guide](tools/README.md) for throughput benchmarking and recovery,
[the publication guide](docs/publication.md) for reproducible comparisons, and
[TEMPLATE.md](TEMPLATE.md) for adapting the task boundary to a new project.

## Repository layout

`envs/` owns simulation; `learning/` owns algorithms and data; `workflows/` has
thin Hydra launchers; `configs/` contains the versioned experiment recipe.
`data/` and `runs/` hold ignored generated artifacts. See
[CONTRIBUTING.md](CONTRIBUTING.md) for verification commands.

## Independent research methods

This repository also hosts isolated implementations of new papers, with shared
simulation contracts and evaluation tools. The first additional track is
[SuperMimic-style learned dynamics and analytic policy gradients](docs/methods/supermimic.md),
including simulator/model PPO comparisons, LEAP hand benchmarks, and separate
visual DAgger and direct RGB experiments. Preview it with:

```bash
uv run --no-sync python -m workflows.train_supermimic
```

See the [paper index](notes/README.md). Each method must establish its own learning
results; the canonical DRLR/RMA recipe remains unchanged.
