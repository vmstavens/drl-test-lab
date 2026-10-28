# Research template

This checkout is an independent research template. Its supported learning task
is WorkbenchReach, not the parent insertion project's force task. Read README.md
and docs/canonical-learning-setup.md before changing learning contracts.

- The sole phase-one recipe is configs/train_recursive_drlr_rma.yaml. Keep RMA
  history adaptation separate in configs/train_recursive_rma_phase2.yaml.
- Preserve DP bootstrap, accepted-parent recursion, successful-reference replay
  and disjoint learner-only evaluation gates. Assisted training success is not
  a learner evaluation.
- Previous action always means the normalized command actually executed. Keep
  terminal/reset history zero. Stateful deployment policies receive executed
  commands through observe_transition(); never assume a request was executed.
- Frames, units, timing, preprocessing, factor layout, history and normalization
  are compatibility contracts. Reject incompatible checkpoints rather than
  adding automatic legacy conversion trees.
- Keep one Task boundary, pure algorithm code in learning/, task physics in
  envs/, experiment settings in configs/ and thin launchers in workflows/.
- Use locked Python 3.11. Select JAX CUDA packages manually and preserve them
  with --inexact/--no-sync. No sibling repository is a runtime dependency.
- New launchers default to dry-run. Never commit data, weights, run outputs,
  credentials or absolute local checkpoint defaults.
- Run ruff and the CPU suite; add meaningful contract regressions for learning
  changes. Document intentional differences from the saved upstream evidence.
- The worked example uses 0.5 ms physics and 2 ms control. A different research
  task may declare other explicit timing through its versioned contract.
