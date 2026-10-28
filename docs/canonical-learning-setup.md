# Canonical research learning setup

The sole phase-one recipe is `configs/train_recursive_drlr_rma.yaml`.
`configs/train_drlr2.yaml` is an alias. History adaptation is a separate recipe
in `configs/train_recursive_rma_phase2.yaml`. All launchers default to dry-run.

## Source evidence and retained settings

The implementation was extracted and re-expressed from `dlo-insertion-lab`'s
corrected learning path, whose upstream source is `dlo-manipulation-lab` commit
`d739e9360280618ba34825dd03e9dcbbb3dfc0b5`. The saved successful stage configuration
and lineage evidence are preserved in `canonical-learning-provenance.json`.
Transfer file hashes are in `template-transfer.json`.

The source recorded 490/512 phase-one successes and 471/512 after history
adaptation in its insertion lineage. Those are **not template results**. They
also do not demonstrate one unchanged fresh-DP-to-full-domain training run.
The worked example and newly expressed implementation require their own
learning validation; passing regression tests is not convergence evidence.

The default preserves these source learning settings, checked automatically:

| Setting | Value |
|---|---|
| Environments / online replay capacity | 128 / 1,000,000 |
| Minibatch / post-warmup updates per vector step | 512 / 2 |
| Critic-only warmup | 4,000 vector steps, one critic update per vector step |
| Successful reference replay | 30%, online replay 70% |
| Actor / critic learning rate | 1e-5 / 3e-4 |
| Actor / critic hidden widths | [256,256] / [512,512,256,256,128] |
| Critic layer normalization | enabled |
| Discount / Polyak / Bellman target clipping | 0.9995 / 0.005 / [-15,100] |
| Policy std initial / minimum / maximum | 0.02 / 0.005 / 0.04 |
| Entropy temperature | fixed 0.001; policy mean/std still learn |
| Action EMA | disabled; previous executed command remains observable |
| Stage budget | 100M transitions |

During critic-only warmup, Bellman backup always uses the frozen reference
and actor/entropy-temperature updates remain frozen. After warmup, both target
critics must strictly prefer the learner to select it; ties select the reference. Bellman backup uses the selected action and applies entropy
only when the learner is selected. SAC actor loss, twin-Q MSE and Polyak target
updates remain. No BC anchor, trust region or experimental actor objective is
included. Observation normalization is frozen from the DP's training-only
statistics, with scale floor 0.001. Privileged factors have fixed task-defined
coordinates; previous commands stay in normalized action space.

## Explicit differences and supported scope

- **Task:** a small MuJoCo XY reaching task replaces insertion for the worked
  example: four observable position/velocity values, two force commands and two
  hidden actuator/load factors. Physics is 0.5 ms, control is 2 ms. Sparse reward
  remains -0.01 per step and +100 success bonus. This is not a force-sensing DLO
  port or a claim that the source's hyperparameters are optimal for reaching.
- **Structure:** the active algorithm is expressed as small Flax/JAX modules,
  with one `Task` boundary. Historical RMA layouts, cross-frame conversions,
  checkpoint transplantation, skrl/Brax backend dispatch and unrelated learning
  experiments are deliberately excluded. Unused scaffold trainer/transform
  controls are removed rather than silently ignored. Old insertion checkpoint formats are
  not compatible with the new semantic contract. Network parameter names and
  numerical execution differ, even where the equations/settings match.
- **Controller correctness:** previous action is the actual executed normalized
  command for learner/reference inputs and replay; it is zero on terminal/reset
  boundaries. This intentionally fixes the upstream candidate-history bug.
  Deployment receives executed-action feedback instead of assuming requested
  actions were executed. No action EMA or motion limiter is added.
- **DP:** temporal U-Net, two-observation history, 16-step prediction, four
  inference diffusion steps and one receding-horizon command per control step.
  Training uses the upstream policy optimizer and training-only min/max
  observation statistics. An explicit per-update weight EMA (0.995 default) is
  used for exported inference parameters. No force-specific preprocessing or
  historical checkpoint adaptation is included.
- **Domain curriculum:** normalized gain/load envelopes widen from nominal
  through the configured 0-to-1 levels. Physical distributions and reset geometry
  differ from the insertion lineage. Only an accepted parent initializes the
  next stage and supplies its frozen reference. Critics/optimizers start fresh
  for a new domain; recovery restores in-flight state instead.
- **Evaluation scheduling:** retains the insertion project's later optimization:
  100 progress episodes every 5M transitions, first at 5M; a passing screen
  triggers 512 promotion episodes on disjoint seeds. Promotion requires >=95%.
  Source checks were every 500k transitions. An inherited parent receives a full
  initial learner-only check and can pass without training in the new domain.
  Evaluations and collection use rolling slots. Training successes are labeled
  assisted and never used as acceptance evidence.
- **Reference replay:** bootstrap requires >=40 successes and >=80% of 64 fresh
  DP attempts. Accepted actors collect on separate training seeds: 512 attempts,
  >=400 successes. Successful replay carries forward, capped to the most recent
  1M transitions to keep memory bounded. Validation/promotion seeds are rejected
  if they overlap reference-collection seed ranges.
- **Logging:** bounded chunks write assisted episode counts and mean losses;
  actor checkpoints/recovery are requested every 100k transitions. Chunk
  boundaries can overshoot scheduling thresholds. Evaluation gates retain
  their configured episode counts and thresholds.
- **Execution:** both split and fused execution use the same core step/update
  functions. Fused is the template default, including DP; it is configurable for
  benchmarking. This differs from insertion's opt-in DP fusion default and can
  change floating-point results. First compilation is reported separately from
  warmed throughput; no speedup is assumed on a different task or device.
- **RMA:** the actor and privileged encoder freeze. The upstream adaptation
  encoder, 25-round DAgger schedule, 64 environments, batch 512, 2,000 updates per
  round, 1e-4 learning rate, 10 history entries and 0.1 s spacing remain. At 2 ms
  control the spacing is 50 controls. History clocks reset per episode. Only
  student history conditioning is evaluated; failed bundles refuse deployment.
- **Recovery:** phase-one archives include simulator, replay, RNG, learner,
  target critics and optimizer state, plus evaluation/checkpoint scheduling and
  reference hashes. Resume writes a new run; source files remain untouched.
  RMA round state is not currently recoverable; restart phase two from the
  accepted phase-one artifact.

The baseline dependency stack is Python 3.11, JAX 0.10.2, MuJoCo/MJX 3.11 and
pinned `mjxsim`/`mjsim` revisions from the tested insertion stack. CUDA JAX extras
are selected manually. Package locks, configs and run provenance record actual
versions; the template does not load code from another checkout.
