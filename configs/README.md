# Experiment configuration

`train_recursive_drlr_rma.yaml` is the sole phase-one research recipe.
`train_drlr2.yaml` aliases it. `train_recursive_rma_phase2.yaml` runs separate
history adaptation from an accepted phase-one stage. Old placeholder experiment
presets were removed; experiment overrides are recorded with each run.

The default task is `workbench_reach`. It declares 0.5 ms physics and 2 ms
control. `contract/workbench_reach.yaml` supplies observable/privileged layouts,
action bounds and units, coordinate frame, preprocessing and executed-history
semantics. Changing these requires compatible datasets and checkpoints.

```bash
uv run --no-sync python -m workflows.collect_demos
uv run --no-sync python -m workflows.prepare_dataset
uv run --no-sync python -m workflows.train_dp
uv run --no-sync python -m workflows.train_recursive_drlr_rma
uv run --no-sync python -m workflows.train_recursive_rma_phase2
```

These commands only compose and validate configuration by default. Supply
`runtime.dry_run=false` to execute. DP must be trained before supplying
`il.checkpoint=.../models/final_model.pkl`. RMA requires
`phase1_job=.../stage_020` (or another accepted stage).

`runtime.device=cpu` selects CPU learning; `runtime.device=cuda` selects the
manually installed JAX GPU backend. Do not let an automatic sync replace GPU
packages: use `uv run --no-sync` and `uv sync --frozen --inexact`.

The RL budget explicitly counts transitions; warmup counts vector steps.
Network and optimizer defaults, gate thresholds, source evidence and every
intentional source difference are documented in
[the canonical setup](../docs/canonical-learning-setup.md).

For a bounded implementation experiment, use Hydra overrides rather than a new
preset, for example `rl.num_envs=4 rl.budget.value=1024`. Such an experiment is
not a full learning validation and must retain its recorded overrides. Do not
silently weaken bootstrap or learner acceptance gates to make a run pass.

Optional TensorBoard settings use `runtime.tensorboard.*`. Optional ntfy
notifications use `runtime.notifications.*`; topic/token values come from
environment variables, never committed config. Notifications are disabled by
default. Existing publication matrices/protocols remain usable with the new
learning workflows; see [the publication guide](../docs/publication.md).
