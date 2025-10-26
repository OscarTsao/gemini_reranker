"""MLflow helpers with resilient logging and rich metadata snapshots."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import mlflow
import torch
from omegaconf import OmegaConf


LOGGER = logging.getLogger(__name__)


def _git_metadata() -> dict[str, str]:
    """Return Git commit metadata if available."""

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, env=env)
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover - git not present
        return {}

    try:
        status = (
            subprocess.check_output(["git", "status", "--short"], cwd=repo_root, env=env)
            .decode("utf-8")
            .strip()
        )
        dirty = "clean" if not status else "dirty"
    except Exception:
        dirty = "unknown"
    return {"git_sha": sha, "git_status": dirty}


@contextlib.contextmanager
def get_or_create_run(cfg, resolved_cfg: Optional[dict] = None) -> Iterator[mlflow.ActiveRun]:
    """Context manager that configures MLflow and logs core metadata."""

    tracking_uri = cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run = mlflow.start_run(run_name=cfg.mlflow.run_name)
    try:
        safe_set_tags(
            {
                "project": cfg.project_name,
                "job_name": cfg.mlflow.run_name,
                **_git_metadata(),
            }
        )
        if resolved_cfg is None:
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        log_config_yaml(resolved_cfg)
        log_hardware_snapshot()
        yield run
    finally:
        mlflow.end_run()


def log_config_yaml(config_dict: dict) -> None:
    """Persist the resolved Hydra config as a YAML artifact."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "config.yaml"
        conf = OmegaConf.create(config_dict)
        tmp_path.write_text(OmegaConf.to_yaml(conf), encoding="utf-8")
        log_artifact_dir(tmp_path.parent, artifact_path="config")


def log_hardware_snapshot() -> None:
    """Log hardware and software metadata to MLflow."""

    cuda_available = torch.cuda.is_available()
    snapshot = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "torch_cuda_arch": torch.cuda.get_device_capability(0) if cuda_available else None,
        "amp_enabled": torch.cuda.is_available(),
    }
    safe_log_dict(snapshot, artifact_name="env/hardware_snapshot.json")


def log_dataset_card(card: dict[str, object], dataset_name: str) -> None:
    """Persist dataset metadata to MLflow as JSON."""

    safe_log_dict(card, artifact_name=f"datasets/{dataset_name}_card.json")


def log_model_summary(summary: str, name: str = "model_summary.txt") -> None:
    """Store a textual model summary for later inspection."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_text(summary, encoding="utf-8")
        log_artifact_dir(path.parent, artifact_path="model")


def safe_log_dict(payload: dict[str, object], artifact_name: str) -> None:
    """Serialize and log a dictionary payload as a JSON artifact."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / Path(artifact_name).name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifact_path = Path(artifact_name).parent.as_posix().strip(".")
        if artifact_path and artifact_path != ".":
            log_artifact_dir(path.parent, artifact_path=artifact_path)
        else:
            log_artifact_dir(path.parent)


def log_metrics(step: Optional[int] = None, **metrics: float) -> None:
    """Log metrics safely without failing the training loop."""

    if not metrics:
        return
    sanitized = {}
    for key, value in metrics.items():
        safe_key = (
            key.replace("@", "_at_")
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
        )
        sanitized[safe_key] = float(value)
    try:
        mlflow.log_metrics(sanitized, step=step)
    except Exception as exc:  # pragma: no cover - network issues
        LOGGER.warning("Failed to log MLflow metrics %s: %s", sanitized, exc)


def log_artifact_dir(path: Path | str, artifact_path: Optional[str] = None) -> None:
    """Upload a directory or single file to MLflow artifacts."""

    path = Path(path)
    if path.is_file():
        parent = path.parent
    else:
        parent = path
    try:
        mlflow.log_artifacts(parent.as_posix(), artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover - network issues
        LOGGER.warning("Failed to log MLflow artifacts from %s: %s", parent, exc)


def safe_set_tags(tags: dict[str, object]) -> None:
    """Set MLflow tags, swallowing transient errors."""

    if not tags:
        return
    try:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to set MLflow tags %s: %s", tags, exc)
