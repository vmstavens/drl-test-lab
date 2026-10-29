# Learning layer

The supported learning path is implemented: scripted demonstration collection,
immutable transition preparation, temporal U-Net DP, recursive DRLR/SAC with
successful-reference replay, and separate RMA history adaptation.

- `contracts.py`: semantic frames, layouts, action units/bounds, preprocessing,
  timing and executed-history compatibility.
- `collection/`: successful scripted rollouts through the shared controller.
- `data/`: hashed manifests, episode-delimited arrays and lazy sequence splits.
- `imitation/`: DP training, frozen normalization and explicit checkpoint loader.
- `reinforcement/task.py`: the one normalized-command/batched-task boundary.
- `reinforcement/drlr2.py`: actor, twin critics, replay, step and gradient kernels;
  fused/split execution share those same functions.
- `reinforcement/recursive.py`: bootstrap, accepted-parent recursion, disjoint
  learner-only gates, successful handoff and phase-one recovery.
- `reinforcement/rma.py`: frozen privileged policy and separate history encoder
  training; deployment evaluation uses the student alone.
- `policies/`: a small reset/act/transition-feedback interface and concrete DP/RMA
  loaders. No implicit legacy checkpoint adaptation.
- `evaluation/`, `rendering/`: shared generic evaluation, viewing and video loop.
- `publication/`, `tensorboard.py`, `notifications/`, `config/`: existing research
  lifecycle, analysis, monitoring and provenance tools.

`workflows -> learning -> envs` is the dependency direction. Task classes never
import learning algorithms. See ../TEMPLATE.md for the explicit task protocol
and ../docs/canonical-learning-setup.md for source evidence and adaptations.

`supermimic/` is an independent research track with autoregressive learned dynamics,
BPTT actor updates, simulator/model PPO controls, held-out diagnostics, strict
actor/model bundles and visual experiments. It uses the same Task boundary and
leaves the canonical recursive learner unchanged. See
[the method guide](../docs/methods/supermimic.md).
