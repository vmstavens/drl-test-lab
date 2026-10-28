# Starting a research project

Start by running the small `WorkbenchReach` example. It is the supported
end-to-end learning example; DLO insertion and the Cartpole policy-loader demo
remain available for simulation and tooling exploration.

1. Replace the project identity and ownership in `pyproject.toml`, README and
   license. Keep Python/dependency choices explicit and regenerate `uv.lock`
   only when intentionally changing them.
2. Add your task under `envs/` and register it. Keep physics independent of the
   learning package. Supply `reset`, `step`, `observation_layout`,
   `learning_contract`, `info['privileged']`, `info['steps']` and
   `info['executed_action']` (physical commands after clipping/filtering).
   For scripted collection also supply `expert(observation)`.
3. Declare a task config and one contract: observation/factor layout, coordinate
   frame, action bounds and units, control and simulation intervals,
   preprocessing identity and executed-history semantics.
4. Set the task's privileged dimension in the canonical recipe. The single
   `learning/reinforcement/task.py` boundary batches simulation and normalizes
   commands. Do not build historical observation/checkpoint conversion trees.
5. Collect, prepare and train DP. Dataset splits are by episode; normalization
   uses only training sequences. Never normalize against validation/test data.
6. Validate the frozen DP and bootstrap gate before running recursive DRLR.
   Evaluate the learner alone. Preserve successful reference replay and accepted
   parent recursion. A failed gate is a failed experiment, not a reason to
   silently weaken the threshold.
7. Run RMA history adaptation separately, using an accepted phase-one actor.
   Only the history adapter learns. Keep history spacing in physical seconds
   explicit when changing control timing.
8. Remove unused example assets/configs after the replacement works. Keep
   algorithms in `learning/`, simulation in `envs/`, experiments in `configs/`
   and thin launchers in `workflows/`.

Stateful deployment policies implement `observe_transition(executed_action)`;
the shared evaluation/video runner supplies the physical command actually
executed. `reset(seed=...)` must clear all queues, histories and episode state.

Keep one canonical recipe rather than accumulating alternative defaults. Small
experiments use explicit Hydra overrides recorded with the run. Document task,
architecture and scheduling differences from the saved source recipe before
claiming a faithful reproduction or comparing success rates.

## Clean-clone verification

```bash
uv sync --frozen --inexact
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
uv run --no-sync python -m workflows.collect_demos
uv run --no-sync python -m workflows.prepare_dataset
uv run --no-sync python -m workflows.train_dp
uv run --no-sync python -m workflows.train_recursive_drlr_rma
uv run --no-sync python -m workflows.train_recursive_rma_phase2
git status --short
```

Generated datasets, checkpoints and run outputs are ignored. Portable tests
check code/contracts; a full-budget learning campaign is a separate experiment.
Existing matrix, analysis, plotting and reproduction tools are described in
[docs/publication.md](docs/publication.md).
