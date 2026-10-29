"""Strict world-model bundles for controlled frozen-model optimizer studies."""

import hashlib
import json
from pathlib import Path

import flax.serialization
import jax


def model_contract(trainer):
    options = trainer.options
    return dict(
        schema_version=1,
        physical_contract=trainer.task.contract.to_dict(),
        model_spec=trainer.spec,
        model_widths=options["model_widths"],
        critic_widths=options["critic_widths"],
        reward_scale=options["reward_scale"],
        observation=options["observation"],
        pixel_widths=options["pixel_widths"],
        pixels=options["pixels"] if trainer.visual else None,
        render_spec=trainer.task.base.render_spec if trainer.visual else None,
        outcome_heads="normalized_reward_and_physical_terminal_v1",
    )


def load_world_model(trainer, manifest):
    manifest = Path(manifest).expanduser()
    metadata = json.loads(manifest.read_text())
    if metadata["contract"] != model_contract(trainer):
        raise ValueError("World model architecture or semantic contract differs")
    payload = (manifest.parent / metadata["checkpoint"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("World-model checkpoint hash differs from manifest")
    template = dict(
        model=trainer.parameters.model,
        critic=trainer.parameters.critic,
        pixels=trainer.parameters.pixels,
    )
    loaded = flax.serialization.from_bytes(template, payload)
    for expected, actual in zip(
        jax.tree.leaves(template), jax.tree.leaves(loaded), strict=True
    ):
        if expected.shape != actual.shape:
            raise ValueError("World-model parameter layout differs")
    return trainer.parameters.replace(
        model=loaded["model"],
        pixels=loaded["pixels"],
        model_opt=trainer.optimizer.init(loaded["model"]),
        pixels_opt=trainer.optimizer.init(loaded["pixels"]),
    )
