"""Typed project-level schemas registered with Hydra's ConfigStore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TaskConfig:
    name: str = MISSING
    registration_modules: list[str] = field(
        default_factory=lambda: ["envs.grasped_end_dlo"]
    )
    env: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetExpectedConfig:
    observation_dim: int = MISSING
    action_dim: int = MISSING
    control_dt: float = MISSING


@dataclass
class DatasetConfig:
    id: str = MISSING
    kind: str = MISSING
    path: str = MISSING
    manifest: str = MISSING
    expected: DatasetExpectedConfig = field(default_factory=DatasetExpectedConfig)


@dataclass
class CollectorConfig:
    name: str = MISSING
    controller: str = MISSING
    episodes: int = 10
    max_steps: int = 1000
    batch_size: int = 16


@dataclass
class TransformConfig:
    name: str = MISSING
    output_path: str = "data/workbench_reach/processed/v1"
    dataset_id: str = "workbench_reach_processed_v1"


@dataclass
class DiffusionPolicyConfig:
    name: str = "diffusion_policy"
    checkpoint: str | None = None
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    diffusion_step_embed_dim: int = 256
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    validation_fraction: float = 0.2
    split_seed: int = 42
    weight_ema_decay: float = 0.995
    max_batches: int | None = None


@dataclass
class DRLR2Config:
    num_envs: int = 128
    memory_size: int = 1000000
    batch_size: int = 512
    gradient_steps: int = 2
    warmup_vector_steps: int = 4000
    discount_factor: float = 0.9995
    polyak: float = 0.005
    actor_learning_rate: float = 1e-05
    critic_learning_rate: float = 0.0003
    bellman_target_min: float = -15.0
    bellman_target_max: float = 100.0
    initial_action_std: float = 0.02
    minimum_action_std: float = 0.005
    maximum_action_std: float = 0.04
    actor_hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_sizes: list[int] = field(
        default_factory=lambda: [512, 512, 256, 256, 128]
    )
    critic_layer_norm: bool = True
    reference_replay_fraction: float = 0.3
    freeze_entropy_temperature: bool = True
    initial_entropy_temperature: float = 0.001
    entropy_learning_rate: float = 0.0003
    previous_action_source: str = "executed"
    name: str = "drlr2_sac"
    budget: dict[str, Any] = field(
        default_factory=lambda: {"unit": "transitions", "value": 100000000}
    )
    encoder_hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    latent_dim: int = 8
    privileged_dim: int = 2
    chunk_steps: int = 64
    checkpoint_interval_transitions: int = 100000
    teacher_diffusion_steps: int = 4
    training_execution: str = "fused"
    evaluation_gate: dict[str, Any] = field(
        default_factory=lambda: {
            "interval_transitions": 5000000,
            "minimum_transitions": 5000000,
            "episodes": 512,
            "progress_episodes": 100,
            "batch_size": 128,
            "success_threshold": 0.95,
            "seed_start": 2026090900,
            "progress_seed_start": 2027090900,
        }
    )
    bootstrap: dict[str, Any] = field(
        default_factory=lambda: {
            "episodes": 64,
            "minimum_successes": 40,
            "minimum_success_rate": 0.8,
            "seed_start": 10000,
        }
    )
    handoff: dict[str, Any] = field(
        default_factory=lambda: {
            "episodes": 512,
            "minimum_successes": 400,
            "seed_start": 2026091900,
        }
    )


@dataclass
class EvaluationConfig:
    protocol: str | None = None
    split: str = "test"
    condition: str = "nominal"
    method: str = "unspecified"
    training_seed: int | None = None
    episodes: int = 20
    deterministic: bool = True
    max_steps: int = 1000
    record_video: bool = False
    seeds: list[int] = field(default_factory=list)
    policy_manifest: str | None = None
    policy: str = "checkpoint"
    compile_environment: bool = True
    fps: int = 30
    width: int = 640
    height: int = 480
    camera: str | None = None
    interactive: bool = False


@dataclass
class TensorBoardConfig:
    mode: str = "print"
    log_dir: str | None = None
    port: int = 6006
    open_browser: bool = False


@dataclass
class NotificationConfig:
    enabled: bool = False
    provider: str = "ntfy"
    server: str = "https://ntfy.sh"
    topic_env: str = "NTFY_TOPIC"
    token_env: str = "NTFY_TOKEN"
    on_success: bool = True
    on_failure: bool = True
    on_interrupted: bool = True
    minimum_success_duration_seconds: float = 60.0
    timeout_seconds: float = 5.0


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    data_root: str = "data"
    runs_root: str = "runs"
    dry_run: bool = True
    provenance_inputs: dict[str, str] = field(default_factory=dict)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)


@dataclass
class CollectWorkflowConfig:
    contract: dict[str, Any] = field(default_factory=dict)
    stage: str = "collect_demos"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class PrepareWorkflowConfig:
    contract: dict[str, Any] = field(default_factory=dict)
    stage: str = "prepare_dataset"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class TrainDPWorkflowConfig:
    contract: dict[str, Any] = field(default_factory=dict)
    stage: str = "train_dp"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    il: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class TrainDRLR2WorkflowConfig:
    recursive: dict[str, Any] = field(default_factory=dict)
    rma: dict[str, Any] = field(default_factory=dict)
    phase1_job: str | None = None
    contract: dict[str, Any] = field(default_factory=dict)
    stage: str = "train_drlr2"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    il: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    rl: DRLR2Config = field(default_factory=DRLR2Config)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class EvaluateWorkflowConfig:
    stage: str = "evaluate"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_REGISTERED = False


def register_configs() -> None:
    """Register workflow schemas once, before Hydra composes YAML configs."""
    global _REGISTERED
    if _REGISTERED:
        return
    store = ConfigStore.instance()
    schemas = {
        "collect": CollectWorkflowConfig,
        "prepare": PrepareWorkflowConfig,
        "train_dp": TrainDPWorkflowConfig,
        "train_drlr2": TrainDRLR2WorkflowConfig,
        "evaluate": EvaluateWorkflowConfig,
    }
    for name, node in schemas.items():
        store.store(
            group="schema",
            name=name,
            node=node,
            package="_global_",
            provider="drl-sys-lab",
        )
    _REGISTERED = True
