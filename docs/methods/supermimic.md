# SuperMimic research track

This is an independent, from-scratch implementation of the learning mechanism in
[SuperMimic](../../notes/super-mimic.pdf), sections 3 and appendices A–D. It is a
research adaptation to reaching and dexterous manipulation. It does not reproduce
the humanoid experiment, its scaling law, GPU-hour reduction or physical transfer.
No convergence or performance advantage has been established by the CPU checks.

## Method isolation

The canonical DP → recursive DRLR → RMA path and its sparse WorkbenchReach task
are unchanged. The new `learning/supermimic/` package implements model fitting,
analytic actor updates, matched PPO comparisons, visual extensions and exports.
Physics and observation definitions live in `envs/supermimic_*.py`. All training
uses the existing `learning/reinforcement/task.py` boundary. Workflows use the
existing configuration, provenance, notifications and evaluation infrastructure.
There is no sibling checkout dependency. Generated assets, data and actors stay
in ignored directories.

| Mode (`supermimic.method`) | Experience and optimization |
| --- | --- |
| `supermimic` | Current-policy simulator data; autoregressive dynamics fitting; BPTT through a frozen learned model; soft survival and a terminal critic |
| `ppo_sim` | Squashed Gaussian PPO on simulator rollouts, with time-limit bootstrapping and episode-safe GAE |
| `ppo_wm` | Same model-fitting procedure; squashed Gaussian PPO on imagined rollouts, weighted by predicted survival |
| `visual_dagger` | A state teacher labels student-visited simulator states; teacher intervention decays to zero; a visual actor fits those labels |

The actor and critic are distinct. The model and critic may see privileged state
and physical context; the state actor sees the declared observation, and the RGB
actor sees only RGB, joint proprioception, the goal and previous executed command.
The RGB actor never receives object pose/velocity or physical context as numeric
features. Model fitting cannot update the actor; actor fitting freezes model
weights while retaining derivatives with respect to model inputs. Critic targets
are computed once per rollout, then frozen across critic epochs.

Replay keeps contiguous windows, never crosses an episode/reset, and partitions
whole environment lanes into training and holdout sets. Holdout states never seed
policy updates or fit models. Collection stores the executed action for dynamics
learning and separately stores the sampled pre-tanh action and likelihood for PPO.
Terminal and reset actor histories are zero. Physical termination labels are
separate from time limits: a timeout ends a replay window and GAE recursion, but
still bootstraps its final physical state.

## Benchmark contracts

| Task/contract configuration | State and task | Timing |
| --- | --- | --- |
| `supermimic_reach` | Dense, fixed-duration variant of WorkbenchReach. Position/velocity state; hidden gain/load context; origin goal. The original sparse task is untouched. | 0.5 ms physics, 2 ms control |
| `supermimic_hand_rotate` | Playground LEAP hand/model/reset distribution; commanded z angular velocity. Success requires one net revolution in the commanded direction. | 10 ms physics, 50 ms control |
| `supermimic_hand_reorient` | Same hand; a fixed quaternion goal per episode. Success means orientation error below 0.2 rad for 0.25 s. | 10 ms physics, 50 ms control |

The hand uses 16 incremental joint commands. Each requested physical increment
is clipped to ±0.5 rad, then the accumulated target is clipped to the actuator
position limits. Feedback is the realized increment in radians; normalized
history is that increment divided by 0.5. Imagined actuator processing uses the
same limits and updates accumulated targets analytically. The world state is:
23 qpos (16 joint angles, object xyz, object wxyz quaternion), 22 qvel (16 joint
velocities and the object's six free-joint velocities), and 16 actuator targets.
Quaternion predictions are projected onto the unit sphere. The state actor sees
qpos/qvel and the goal; previous executed action is appended by the learner.

Hand context is the per-episode cube mass and friction scale. Mass and inertia
scale together. Dense reward combines exponential orientation/velocity tracking,
object-position retention, command/velocity penalties, and a drop penalty; there
is no sparse success bonus. A drop below world z = −0.05 m ends the episode;
success is an evaluation event rather than a reset. Success and subsequent drop
are reported separately. The moving reference goal, rewards, action semantics
and success counters of upstream `LeapCubeReorient` are not reused unchanged.
These are versioned research variants, not scores on an unmodified Playground task.

Each actor export records the physical contract, ordered world-state layout,
frozen declared scales, architecture, observation mode, image dimensions/camera,
asset digest and actor digest. Loading rejects incompatible contracts/assets;
there is no legacy checkpoint conversion. A deployment actor uses
`observe_transition()` feedback and never advances command history in `act()`.

## Running the stages

Use the locked Python 3.11 environment. Preserve manually selected JAX CUDA
packages with `uv sync --frozen --inexact` and `uv run --no-sync`. All commands
below are previews until `runtime.dry_run=false` or an explicit `--run` is added.
Choose a fresh `hydra.run.dir` for every execution.

```bash
# Reaching implementation check; add runtime.dry_run=false to train.
uv run --no-sync python -m workflows.train_supermimic

# Fetch only the pinned Menagerie LEAP assets, into ignored data/.
uv run --no-sync python -m tools.fetch_research_assets
uv run --no-sync python -m tools.fetch_research_assets --run

# Contact check; use runtime.device=cuda with a matching JAX CUDA installation.
uv run --no-sync python -m workflows.train_supermimic \
  task=supermimic_hand_rotate contract=supermimic_hand_rotate supermimic=hand \
  task.env.menagerie_path=data/assets/mujoco_menagerie

# Main state-based benchmark.
uv run --no-sync python -m workflows.train_supermimic \
  task=supermimic_hand_reorient contract=supermimic_hand_reorient supermimic=hand \
  task.env.menagerie_path=data/assets/mujoco_menagerie

# Select either controlled baseline without changing task/observations/reward.
# Append supermimic.method=ppo_sim or supermimic.method=ppo_wm to that command.

# Visual deployment experiment: supply an exported state teacher when executing.
MUJOCO_GL=egl uv run --no-sync python -m workflows.train_supermimic \
  task=supermimic_hand_reorient contract=supermimic_hand_reorient \
  supermimic=visual_dagger \
  task.env.menagerie_path=data/assets/mujoco_menagerie \
  supermimic.teacher_manifest=runs/state_teacher/final/policy.json

# Direct RGB extension, through a learned differentiable image head.
MUJOCO_GL=egl uv run --no-sync python -m workflows.train_supermimic \
  task=supermimic_hand_reorient contract=supermimic_hand_reorient \
  supermimic=visual task.env.menagerie_path=data/assets/mujoco_menagerie
```

`supermimic=hand` fits 4 steps (0.20 s) and imagines 12 (0.60 s), rather than
blindly copying a step count from another control rate. The default reaching
configuration uses the paper's k=8, H=32 step counts, which at 2 ms correspond to
0.016 s and 0.064 s. It is a correctness exercise, not a tuned reaching recipe.
For the no-critic ablation set `supermimic.use_critic=false` on analytic training.
Sweep `window_steps` and `horizon_steps` explicitly, recording their seconds.

For quick CPU checks, reduce environment count, widths, batch and update counts.
A useful finite execution check (not a learning experiment) is:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m workflows.train_supermimic \
  runtime.dry_run=false hydra.run.dir=runs/supermimic_smoke \
  supermimic.iterations=2 supermimic.num_envs=3 supermimic.holdout_envs=1 \
  supermimic.window_steps=2 supermimic.horizon_steps=3 \
  supermimic.warmup_steps=3 supermimic.collect_steps=2 \
  supermimic.batch_size=4 supermimic.model_updates=1 supermimic.critic_epochs=1 \
  supermimic.actor_widths=[8] supermimic.model_widths=[8] \
  supermimic.critic_widths=[8] supermimic.evaluation.episodes=2 \
  task.env.episode_length=6
```

## Vision experiments

DAgger is a visual deployment experiment, not proof of end-to-end visual
SuperMimic learning. Train a useful state teacher first, count its training cost,
then train the visual student. Both teacher manifest and actor bytes are hashed
in run provenance. Student evaluation always uses the student alone.

The direct RGB path fits an additional deterministic state/context/goal-to-image
network from rendered replay frames. The actor receives the predicted image
during imagination; return gradients pass through the frozen image network to
predicted physical state. Real simulator rendering supplies supervision and does
not need to be differentiable. Pixel reconstruction loss is logged separately.
This is a research extension of the paper and may fail on occlusion or produce
misleading visual derivatives even when reconstruction error is low.

Both visual paths use the marked Playground cube and remove the goal ghost from
images. Rendering currently uses a host MuJoCo renderer, including readback, so
rendering cost is included in collection/evaluation time. It is intentionally
simple and is not a claim of a GPU-rendering speedup. RGB observations and replay
can be expensive; the visual preset uses a smaller buffer/batch. Training assumes
an undelayed deterministic image head; sensor delay is an explicit evaluation
condition. Image queues and command history reset between episodes.

## Comparisons and diagnostics

Preview the supplied five-seed × three-method matrices:

```bash
uv run --no-sync python -m tools.run_matrix configs/research/supermimic_reach_matrix.json
uv run --no-sync python -m tools.run_matrix configs/research/supermimic_hand_reorient_matrix.json
```

Add `--run` to execute. The hand matrix selects CUDA explicitly and samples mass
in [0.8,1.2] and friction scale in [0.7,1.3] for every method. These are development
budgets and hyperparameters; tune each baseline reasonably before interpreting
performance. Equal iteration counts do not imply equal simulator or imagination
budgets. Compare both elapsed time and explicit transition budgets.

Training writes `metrics.jsonl`, periodic simulator episode files, `summary.json`
and `final/policy.json`. Logs distinguish collected simulator transitions,
model-fitting transitions, policy imagination transitions and simulator evaluation
transitions. Setup, explicit kernel compilation, first iteration, warmed iteration
average, total time and evaluation/diagnostic time are reported separately.
Time-to-target is total elapsed time at a passing evaluation, including evaluation
and diagnostic cost; `null` means the target was not reached. It is not a promise
of convergence. Model-only inference throughput is not end-to-end training speed.

Periodic progress episodes use seeds starting at 1,000,000 and never contribute
training data. For checkpoint selection and final claims, use the supplied
`configs/research/supermimic_hand_protocol.json`, with disjoint validation seeds
starting at 2,000,000 and test seeds starting at 3,000,000. It defines nominal,
held-out physical combinations, impulse, camera shift, dim lighting, occlusion
and two-frame image-delay conditions. Camera/image conditions affect visual
policies; they are controls for state actors. For example:

```bash
MUJOCO_GL=egl uv run --no-sync python -m workflows.evaluate \
  task=supermimic_hand_reorient \
  task.env.menagerie_path=data/assets/mujoco_menagerie \
  evaluation.policy_manifest=runs/hand/final/policy.json \
  evaluation.method=supermimic evaluation.training_seed=0 \
  evaluation.protocol=configs/research/supermimic_hand_protocol.json \
  evaluation.split=validation evaluation.condition=nominal
```

Select by validation success then return, freeze the actor, and run test once.
Use existing publication analysis on independent training seeds; do not bootstrap
individual evaluation episodes as though they were independent training runs.
Report drop rates, orientation/velocity error, time to success and return alongside
success. A zero-failure or scaling claim would require separate evidence.

Held-out replay diagnostics measure normalized open-loop state error at 1, k and H
steps, reward error and termination BCE. Missing episode-safe long windows are
reported as unavailable, not padded across resets. A separate fresh-reset probe
compares a short model-return gradient direction with simulator finite differences
and logs a model/simulator return gap. This probe is diagnostic: contact gradients
are nonsmooth, and one positive direction does not establish globally useful
Jacobians. The same errors are stratified by contact-activity changes, a slip proxy (object
linear speed above 0.05 m/s with a nonzero cube constraint wrench), and drops. These are
observable diagnostics, not ground-truth contact modes; inspect videos before
interpreting them as regrasp/slip classification.

## Intentional differences and remaining research work

The paper uses humanoid motion libraries, return-based reference reset sampling,
a 10-observation history, large 2048-wide networks, a cosine learning-rate schedule,
4096 environments, k=8 and H=32 at 20 ms control. This track uses goal-conditioned
manipulation/reaching, uniform task resets, current observation plus previous
executed command, smaller configurable networks, a constant learning rate, frozen
declared physical scales, and task-specific horizons/rewards. Reward and physical
termination are learned heads; physical observations are analytic. The hand's
known actuator accumulation is analytic and quaternions are projected. There is
no motion encoder/skill decoder or claim to reproduce motion-library amortization.

PPO uses full-rollout epochs and a Gaussian entropy regularizer before the tanh
transform. Baselines share the task and actor/value widths but need tuning before
a performance comparison. A shared frozen-model optimizer study is supported separately from the online
PPO-WM arm: point both `supermimic` and `ppo_wm` at the same exported
`supermimic.model_manifest=runs/source/final/world_model.json` and set
`supermimic.freeze_model=true`. Both start fresh actors/critics and preserve the
same model bytes. The architecture, outcome scaling, physical state, image and
asset contracts must match. Count the shared model's source training cost when
reporting total cost; no result for this study has been measured. The
existing benchmark protocol is configurable infrastructure, not measured results.

Actor exports are deployable; `training.msgpack` retains dynamics/value/image
parameters for inspection. It is not a crash-resumable trainer snapshot: replay,
simulator state and optimizer state are not exported. The existing recursive
learner's recovery remains intact. Full-budget learning campaigns, contact-regime
analysis, shared frozen-model comparisons, richer observation history, motion-data
scaling and real hardware transfer remain experiments to perform after these
implementation checks. Do not infer their outcomes from passing tests.
