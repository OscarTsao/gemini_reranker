"""Typed configuration schemas validated from Hydra DictConfigs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from pydantic import BaseModel, Field, field_validator


_NOW_PATTERN = re.compile(r"\${now:([^}]+)}")
class RelativePathError(ValueError):
    """Raised when a configuration path is expected to be relative."""

    def __init__(self) -> None:
        super().__init__("path must be relative")


class NumWorkersStringError(ValueError):
    """Raised when a string num_workers override is invalid."""

    def __init__(self) -> None:
        super().__init__('num_workers string must be "auto"')


class NumWorkersNegativeError(ValueError):
    """Raised when num_workers is negative."""

    def __init__(self) -> None:
        super().__init__("num_workers cannot be negative")


def _normalize_string(value: str) -> str:
    """Replace Hydra placeholders that are unavailable outside runtime."""

    def _replace_now(match: re.Match[str]) -> str:
        fmt = match.group(1)
        return datetime.now(tz=timezone.utc).strftime(fmt)

    interpolated = _NOW_PATTERN.sub(_replace_now, value)
    return (
        interpolated.replace("${hydra:job.name}", "hydra_job")
        .replace("${hydra.job.name}", "hydra_job")
    )


class MlflowConfig(BaseModel):
    tracking_uri: str = Field(
        ...,
        description="MLflow tracking URI, e.g. sqlite:///mlflow.db",
    )
    experiment_name: str = Field(
        ...,
        description="Experiment name registered with MLflow.",
    )
    run_name: str | None = Field(
        None,
        description="Readable run identifier for mlflow.start_run; defaults to Hydra job name.",
    )
    artifact_root: Path = Field(
        ...,
        description="Root directory for artifact persistence.",
    )
    autolog: bool = Field(
        True,
        description="Enable MLflow autologging when supported.",
    )

    @classmethod
    @field_validator("artifact_root")
    def _artifact_root_relative(cls, value: Path) -> Path:
        if value.is_absolute():
            raise RelativePathError
        return value


class HardwareConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compile: bool = True
    amp_dtype: Literal["fp16", "bf16", "none"] = "bf16"
    grad_accum_steps: int = Field(1, ge=1)
    gradient_checkpointing: bool = True
    num_workers: str | int = Field("auto", description="Either 'auto' or an explicit worker count.")
    persistent_workers: bool = True
    prefetch_factor: int = Field(4, ge=1)
    pin_memory: bool = True
    tf32: bool = True
    fused_adamw: bool = True
    cudnn_benchmark: bool = True

    @classmethod
    @field_validator("num_workers", mode="before")
    def _normalize_workers(cls, value: str | int) -> str | int:
        if isinstance(value, str):
            lowered = value.lower()
            if lowered != "auto":
                raise NumWorkersStringError
            return lowered
        if value < 0:
            raise NumWorkersNegativeError
        return value


class DataConfig(BaseModel):
    name: str
    path: Path
    split_by_note_hash: bool = True
    max_samples: int | None = None
    cache_dir: Path = Path(".cache/data")
    tokenizer_name: str
    padding: Literal["longest", "max_length"] = "longest"
    truncation: bool = True
    max_length: int = Field(512, ge=8)
    bucketed_batches: bool = True

    @classmethod
    @field_validator("path", "cache_dir")
    def _relative_paths(cls, value: Path) -> Path:
        if value.is_absolute():
            raise RelativePathError
        return value


class ModelConfig(BaseModel):
    model_name: str
    tokenizer_name: str
    from_pretrained_path: Path | None = Field(
        None,
        description="Optional checkpoint directory when warm starting from a local run.",
    )


class TrainConfig(BaseModel):
    task: Literal["criteria_ranker", "evidence_span"]
    lr: float = Field(..., gt=0)
    weight_decay: float = Field(0.0, ge=0.0)
    batch_size_per_device: int = Field(..., gt=0)
    max_steps: int = Field(..., gt=0)
    warmup_ratio: float = Field(0.0, ge=0.0, lt=1.0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    eval_every_steps: int = Field(100, gt=0)
    save_every_steps: int = Field(200, ge=0)
    metric: Literal["map@10", "f1_at_iou"] = "map@10"
    early_stop_patience: int = Field(5, ge=1)
    save_top_k: int = Field(1, ge=0)
    margin: float = Field(0.0, ge=0.0)


class JudgeConfig(BaseModel):
    provider: Literal["mock", "gemini"] = "mock"
    temperature: float = 0.0
    max_tokens: int = 256
    json_mode: bool = True
    batch_size: int = Field(32, gt=0)


class AppConfig(BaseModel):
    model_config = {"extra": "ignore"}

    seed: int = 42
    project_name: str = "gemini_reranker"
    output_dir: Path = Path("outputs/default")
    mlflow: MlflowConfig
    hardware: HardwareConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    judge: JudgeConfig

    @classmethod
    @field_validator("output_dir")
    def _output_relative(cls, value: Path) -> Path:
        if value.is_absolute():
            raise RelativePathError
        return value


def _replace_placeholders(obj: object) -> object:
    if isinstance(obj, dict):
        return {key: _replace_placeholders(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_placeholders(value) for value in obj]
    if isinstance(obj, str):
        return _normalize_string(obj)
    return obj


def parse_config(cfg: DictConfig) -> AppConfig:
    """Validate the Hydra DictConfig into a typed :class:`AppConfig`."""

    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except InterpolationResolutionError:
        container = OmegaConf.to_container(cfg, resolve=False)
        container = _replace_placeholders(container)
        # Attempt resolving again for remaining builtin resolvers.
        container = OmegaConf.to_container(OmegaConf.create(container), resolve=True)

    container = _replace_placeholders(container)
    return AppConfig.model_validate(container)
