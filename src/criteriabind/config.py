"""Configuration dataclasses for Criteria Bind."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class JudgeConfig:
    """Configuration for Gemini judging."""

    model: str = "gemini-2.5-flash"
    response_mime: str = "application/json"
    rubric_version: str = "clinical_rubric_v1"
    safety: dict[str, str] = field(
        default_factory=lambda: {
            "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
            "harassment": "BLOCK_MEDIUM_AND_ABOVE",
            "sexual": "BLOCK_MEDIUM_AND_ABOVE",
            "danger": "BLOCK_MEDIUM_AND_ABOVE",
        }
    )
    max_group_size: int = 5
    retries: int = 3
    backoff_seconds: float = 2.0
    enable_two_pass: bool = True
    drop_on_conflict: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer hyperparameters."""

    lr: float = 2e-5
    weight_decay: float = 0.01
    eps: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(slots=True)
class SchedulerConfig:
    """Scheduler hyperparameters."""

    warmup_steps: int = 100
    total_steps: int | None = None
    num_cycles: float = 0.5


@dataclass(slots=True)
class TrainingConfig:
    """Shared training hyperparameters."""

    model_name_or_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_length: int = 512
    max_grad_norm: float = 1.0
    seed: int = 42
    mixed_precision: str = "bf16"
    resume_from: str | None = None
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    loss_type: str = "ranknet"
    margin: float = 0.3
    mlflow_run_name: str | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses to dict for logging."""
        return asdict(self)


@dataclass(slots=True)
class DataConfig:
    """Data paths used across the pipeline."""

    raw_path: str = "data/raw/redsm5_train.jsonl"
    judging_jobs_path: str = "data/proc/redsm5_judging_jobs.jsonl"
    judged_path: str = "data/judged/redsm5_train.jsonl"
    pairwise_path: str = "data/pairs/redsm5_criteria_train.jsonl"
    dev_path: str | None = "data/pairs/redsm5_criteria_dev.jsonl"
    test_path: str | None = "data/pairs/redsm5_criteria_test.jsonl"


@dataclass(slots=True)
class RunConfig:
    """Top-level configuration aggregator."""

    training: TrainingConfig
    data: DataConfig = field(default_factory=DataConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict(),
            "data": asdict(self.data),
            "judge": self.judge.to_dict(),
        }


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a dictionary.")
    return data


def build_run_config(path: str | Path) -> RunConfig:
    """Construct a :class:`RunConfig` from a YAML file."""
    cfg_dict = load_yaml_config(path)
    training_cfg = TrainingConfig(**cfg_dict["training"])
    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    judge_cfg = JudgeConfig(**cfg_dict.get("judge", {}))
    return RunConfig(training=training_cfg, data=data_cfg, judge=judge_cfg)


def update_dataclass(instance: Any, updates: dict[str, Any]) -> None:
    """Update fields on a dataclass instance in-place."""
    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)


def merge_dicts(*dicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Shallow merge dictionaries left-to-right."""
    merged: dict[str, Any] = {}
    for dct in dicts:
        merged.update(dct)
    return merged
