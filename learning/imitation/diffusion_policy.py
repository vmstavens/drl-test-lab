"""Temporal U-Net DP training with training-only normalization and weight EMA."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax
from mjxsim.agents.action_normalization import ActionNormalization
from mjxsim.agents.jax import DP_CFG, DiffusionPolicy
from skrl.utils.tensorboard import SummaryWriter

from learning.contracts import LearningContract
from learning.data.sequences import load_processed_dataset, make_sequence_splits


def train_diffusion_policy(config, output_dir):
    contract = LearningContract.from_config(config)
    data = load_processed_dataset(config["dataset"]["path"])
    contract.require(data.manifest.observation_preprocessing)
    il = config["il"]
    train, validation = make_sequence_splits(
        data,
        obs_horizon=il["obs_horizon"],
        pred_horizon=il["pred_horizon"],
        validation_fraction=il["validation_fraction"],
        split_seed=il["split_seed"],
    )
    normalization = ActionNormalization.from_bounds(
        contract.action_low,
        contract.action_high,
        contract_id=contract.action_id,
        units=contract.action_units,
    )
    cfg = DP_CFG(
        **{
            key: il[key]
            for key in (
                "obs_horizon",
                "pred_horizon",
                "action_horizon",
                "num_diffusion_iters",
                "batch_size",
                "learning_rate",
                "num_epochs",
                "diffusion_step_embed_dim",
                "down_dims",
            )
        }
    )
    key = jax.random.PRNGKey(config["seed"])
    policy = DiffusionPolicy(
        a_dim=data.manifest.action_dim,
        o_dim=data.manifest.observation_dim,
        config=cfg,
        rng=key,
        action_normalization=normalization,
    )
    # Only observations participating in training sequences may fit statistics.
    indices = np.unique(
        (train.starts[:, None] + np.arange(il["obs_horizon"])).reshape(-1)
    )
    values = data.observations[indices]
    policy.set_stats({"obs": {"min": values.min(axis=0), "max": values.max(axis=0)}})
    optimizer = policy.optimizer
    params = policy.params
    optimizer_state = optimizer.init(params)
    ema = params

    @jax.jit
    def update(params, optimizer_state, ema, batch, key):
        loss, gradient = jax.value_and_grad(policy.loss)(params, batch, key)
        changes, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        params = optax.apply_updates(params, changes)
        ema = jax.tree.map(
            lambda old, new: (
                il["weight_ema_decay"] * old + (1 - il["weight_ema_decay"]) * new
            ),
            ema,
            params,
        )
        return params, optimizer_state, ema, loss

    validate = jax.jit(policy.loss)
    writer = SummaryWriter(str(output_dir / "training"))
    records = []
    try:
        for epoch in range(il["num_epochs"]):
            order = np.random.default_rng(config["seed"] + epoch).permutation(
                len(train)
            )
            losses = []
            for start in range(0, len(order), il["batch_size"]):
                key, step_key = jax.random.split(key)
                params, optimizer_state, ema, loss = update(
                    params,
                    optimizer_state,
                    ema,
                    train.batch(order[start : start + il["batch_size"]]),
                    step_key,
                )
                losses.append(float(loss))
                if il["max_batches"] and len(losses) >= il["max_batches"]:
                    break
            valid_losses = []
            if validation is not None:
                for start in range(0, len(validation), il["batch_size"]):
                    key, step_key = jax.random.split(key)
                    valid_losses.append(
                        float(
                            validate(
                                ema,
                                validation.batch(
                                    np.arange(
                                        start,
                                        min(start + il["batch_size"], len(validation)),
                                    )
                                ),
                                step_key,
                            )
                        )
                    )
                    if il["max_batches"] and len(valid_losses) >= il["max_batches"]:
                        break
            record = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": float(np.mean(valid_losses))
                if valid_losses
                else None,
            }
            if not np.isfinite(record["train_loss"]):
                raise FloatingPointError("Nonfinite DP training loss")
            records.append(record)
            writer.add_scalar(
                tag="DP / Train loss", value=record["train_loss"], timestep=epoch + 1
            )
            if valid_losses:
                writer.add_scalar(
                    tag="DP / Validation loss",
                    value=record["validation_loss"],
                    timestep=epoch + 1,
                )
            print(record, flush=True)
    finally:
        writer.close()
    policy.params = params
    policy.ema.shadow_params = jax.device_get(ema)
    models = output_dir / "models"
    models.mkdir(exist_ok=True)
    checkpoint = models / "final_model.pkl"
    policy.save(checkpoint)
    manifest = dict(
        schema_version=1,
        policy_type="diffusion_policy",
        loader="learning.policies.research:load_dp_policy",
        checkpoint="models/final_model.pkl",
        observation_dim=policy.o_dim,
        action_dim=policy.a_dim,
        action_semantics=contract.action_id,
        config=dict(contract=contract.to_dict(), inference_steps=4),
    )
    (output_dir / "policy.json").write_text(json.dumps(manifest, indent=2))
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {"contract": contract.to_dict(), "training": dict(il), "history": records},
            indent=2,
        )
    )
    (output_dir / "training_metrics.json").write_text(json.dumps(records, indent=2))


class DPReference:
    """Frozen DP candidate, sampled once per control step with explicit history."""

    def __init__(self, checkpoint, contract, *, inference_steps=4):
        path = Path(checkpoint)
        contract.require(json.loads(path.with_suffix(".json").read_text())["contract"])
        self.fingerprint = hashlib.sha256(
            path.read_bytes() + path.with_suffix(".json").read_bytes()
        ).hexdigest()
        self.policy = DiffusionPolicy.from_checkpoint(path)
        normalization = self.policy.action_normalization
        if (
            normalization is None
            or tuple(normalization.low) != contract.action_low
            or tuple(normalization.high) != contract.action_high
            or tuple(normalization.units) != contract.action_units
        ):
            raise ValueError("DP action bounds or units differ from the task contract")
        self.inference_steps = inference_steps
        if self.policy.config.obs_horizon != 2:
            raise ValueError("The supported recursive DP history is two observations")
        if self.policy.o_dim != len(
            contract.observation_layout
        ) or self.policy.a_dim != len(contract.action_low):
            raise ValueError("DP dimensions differ from the task contract")
        if self.policy.action_normalization.contract_id != contract.action_id:
            raise ValueError("DP action normalization differs from task contract")
        self.center = jp.asarray(
            (self.policy.stats["obs"]["max"] + self.policy.stats["obs"]["min"]) / 2
        )
        self.scale = jp.asarray(
            np.maximum(
                (self.policy.stats["obs"]["max"] - self.policy.stats["obs"]["min"]) / 2,
                0.001,
            )
        )

    def __call__(
        self, observation, previous_observation, previous_action, factors, key
    ):
        history = jp.stack((previous_observation, observation), axis=1)
        sequence = self.policy.act(
            history, rng=key, num_steps=self.inference_steps, output_domain="normalized"
        )
        return sequence[:, 0]
