"""Compare real fused/split training updates from identical state; dry-run default."""

import argparse
import copy
import json
import time
from pathlib import Path


def benchmark(config, output, *, steps=32, repeats=3, reference_replay=None):
    import jax

    backend = "cpu" if config["runtime"]["device"] == "cpu" else "gpu"
    with jax.default_device(jax.devices(backend)[0]):
        return _benchmark(
            config,
            output,
            steps=steps,
            repeats=repeats,
            reference_replay=reference_replay,
        )


def _benchmark(config, output, *, steps=32, repeats=3, reference_replay=None):
    import jax
    import numpy as np

    from learning.imitation.diffusion_policy import DPReference
    from learning.reinforcement.drlr2 import Trainer
    from learning.reinforcement.recursive import collect_bootstrap
    from learning.reinforcement.task import Task

    task = Task(config)
    try:
        reference = DPReference(
            config["il"]["checkpoint"],
            task.contract,
            inference_steps=config["rl"]["teacher_diffusion_steps"],
        )
        trainers = {}
        for mode in ("split", "fused"):
            values = copy.deepcopy(config)
            values["rl"]["training_execution"] = mode
            trainers[mode] = Trainer(
                values, task, reference, center=reference.center, scale=reference.scale
            )
        trainer = trainers["fused"]
        if reference_replay is None:
            _, replay = collect_bootstrap(trainer)
        else:
            with np.load(reference_replay, allow_pickle=False) as arrays:
                import jax.numpy as jp

                replay = {key: jp.asarray(arrays[key]) for key in arrays.files}
            if replay["observation"].shape[-1] != trainer.width:
                raise ValueError("Benchmark replay observation width differs")
        state = trainer.initialize()
        # Seed valid online data and bypass warmup only for the timed benchmark.
        for _ in range(2):
            state, _ = trainer._step(state)
        state = state.replace(
            vector_steps=state.vector_steps * 0 + trainer.rl["warmup_vector_steps"] + 1
        )
        records = {mode: [] for mode in trainers}
        compilation = {}
        following = {}
        for mode, current in trainers.items():
            start = time.perf_counter()
            result = current.train_chunk(state, replay, steps)
            jax.block_until_ready(result)
            compilation[mode] = time.perf_counter() - start
        for repeat in range(repeats):
            for mode in ("split", "fused") if repeat % 2 == 0 else ("fused", "split"):
                start = time.perf_counter()
                following[mode], metrics = trainers[mode].train_chunk(
                    state, replay, steps
                )
                jax.block_until_ready((following[mode], metrics))
                seconds = time.perf_counter() - start
                assert (
                    int(following[mode].updates - state.updates)
                    == steps * trainer.rl["gradient_steps"]
                )
                records[mode].append(seconds)
        difference = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(
                jax.tree.leaves(following["split"].policy_params),
                jax.tree.leaves(following["fused"].policy_params),
                strict=True,
            )
        )
        from learning.publication.provenance import source_identity

        report = dict(
            source_sha256=source_identity(Path(__file__).resolve().parents[1]),
            device=str(state.rng.device),
            device_kind=state.rng.device.device_kind,
            backend=state.rng.device.platform,
            jax_version=jax.__version__,
            reference_fingerprint=reference.fingerprint,
            steps=steps,
            repeats=repeats,
            compile_and_first_chunk_seconds=compilation,
            median_seconds={
                mode: float(np.median(values)) for mode, values in records.items()
            },
            split_over_fused=float(
                np.median(records["split"]) / np.median(records["fused"])
            ),
            actor_max_abs_difference=difference,
            transitions_per_second={
                mode: steps * trainer.rl["num_envs"] / float(np.median(values))
                for mode, values in records.items()
            },
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2))
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return report
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Saved stage_config.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--reference-replay",
        type=Path,
        help="Successful stage replay, used only for this execution benchmark",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if not args.execute:
        print("DRY RUN:", vars(args))
        return
    benchmark(
        json.loads(args.config.read_text()),
        args.output,
        steps=args.steps,
        repeats=args.repeats,
        reference_replay=args.reference_replay,
    )


if __name__ == "__main__":
    main()
