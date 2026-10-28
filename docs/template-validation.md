# Template implementation validation

Validation performed on 2026-10-25 using the template's independently synced
Python 3.11 environment and committed lock. These are implementation checks,
not learning-convergence results.

- 87 tests passed across portable/configuration, real CPU MuJoCo simulation
  and compiled-learning checks.
- Fused/split equivalence through critic warmup and post-warmup updates, including
  update counts, parameters and loss metrics. Critic warmup backs up the frozen
  reference even when the untrained critics prefer the learner.
- Executed-command history, terminal/reset masking, fixed entropy temperature,
  teacher-free learner evaluation and observable-only student inference.
- Real simulation/replay/RNG/optimizer recovery round-trip and interrupted-loop
  scheduling; accepted-parent/replay handoff with explicit unit gate fixtures.
- Actual RMA collection and adaptation optimizer updates, frozen actor bytes,
  export/reload and executed-action deployment feedback. The export-path unit
  test uses an explicit evaluation fixture; it is not measured student success.
- Immutable episode preparation, disjoint sequence splits and corruption checks.
- All existing publication, notification, TensorBoard, registry and simulation
  tests remain in the suite. The insertion repository's 17 baseline tests passed
  separately; its code and active environment were not changed by this work.

## Real bounded workflow smoke checks

Using fresh paths under `/tmp`, not committed run artifacts:

1. Scripted collection: 8/8 successful reaching episodes; mean 372 controls,
   or 0.744 seconds of simulated time at 2 ms control.
2. Preparation: successful episodes written as hashed, episode-delimited arrays.
3. DP: two epochs, two batches per epoch, batch 8 and small [8,16] U-Net channels.
   Checkpoint/normalization/contract export completed and loaded through the
   generic evaluation and MP4 workflows. Such a tiny run does not train a useful
   DP; its shortened real bootstrap check correctly rejected 0/4 successes.
4. The throughput tool exercised the actual small DP, MuJoCo simulation and two
   gradient updates per vector step, with identical initial state for split and
   fused modes. Four-step, two-repeat CPU smoke measurements were approximately
   436 versus 645 transitions/s (1.48x), with actor parameter difference about
   7.3e-12. These tiny-task CPU timings do not predict insertion/GPU throughput.
5. The worked example loads directly from local XML. The registry regression
   forbids downloading Menagerie for project-local tasks; built-in Playground
   tasks retain their upstream loading behavior.

Full-budget DP/RL convergence, curriculum success and student deployment success
have not been measured. GPU throughput needs a separate device-specific study.
Phase-one recovery is implemented; phase-two partial-round recovery is not.
