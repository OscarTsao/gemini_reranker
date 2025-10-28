"""Judge CLI supporting mock and Gemini backends."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Callable, Iterable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from ..config_schemas import AppConfig, parse_config
from ..hydra_utils import set_global_seed
from ..io_utils import read_jsonl, write_jsonl
from ..judge.gemini import GeminiJudge, GeminiMissingDependencyError
from ..judge.mock_judge import MockJudge
from ..schemas import JudgingJob, Judgment


LOGGER = logging.getLogger(__name__)


def _resolve_root_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        base = Path(get_original_cwd())
    except Exception:  # pragma: no cover - hydra not initialised
        base = Path.cwd()
    return (base / path).resolve()


def _format_with_split(path: Path, split: str) -> Path:
    return Path(str(path).replace("{split}", split))


def _load_jobs(path: Path) -> list[JudgingJob]:
    return [JudgingJob.model_validate(row) for row in read_jsonl(path)]


def _aggregate_metrics(judgments: Iterable[Judgment]) -> dict[str, float]:
    judgments = list(judgments)
    latencies = [j.latency_s for j in judgments if j.latency_s is not None]
    total_preferences = sum(len(j.preferences) for j in judgments)
    input_tokens = sum(int(j.token_usage.get("input_tokens", 0)) for j in judgments)
    output_tokens = sum(int(j.token_usage.get("output_tokens", 0)) for j in judgments)
    metrics: dict[str, float] = {
        "num_jobs": float(len(judgments)),
        "num_preferences": float(total_preferences),
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
    }
    if latencies:
        metrics["avg_latency_s"] = sum(latencies) / len(latencies)
        metrics["max_latency_s"] = max(latencies)
        metrics["min_latency_s"] = min(latencies)
    return metrics


def _write_metrics(path: Path, split: str, metrics: dict[str, object]) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split"] + sorted(metrics.keys())
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"split": split}
        row.update(metrics)
        writer.writerow(row)


def _build_provider(app_cfg: AppConfig) -> tuple[Callable[[Iterable[JudgingJob]], list[Judgment]], str, str]:
    provider = app_cfg.judge.provider
    if provider == "mock":
        judge = MockJudge(
            temperature=float(app_cfg.judge.temperature),
            model_name=app_cfg.judge.model or "offline-mock-v1",
        )
        return judge.batch, provider, judge.model_name

    if provider == "gemini":
        api_key = app_cfg.judge.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "judge.provider=gemini requires GEMINI_API_KEY to be set in the environment "
                "or provided via judge.api_key."
            )
        try:
            judge = GeminiJudge(
                api_key=api_key,
                model=app_cfg.judge.model or "gemini-2.5-flash",
                temperature=float(app_cfg.judge.temperature),
                top_p=float(app_cfg.judge.top_p),
                max_output_tokens=int(app_cfg.judge.max_output_tokens),
                json_mode=bool(app_cfg.judge.json_mode),
                timeout_s=int(app_cfg.judge.timeout_s),
                max_retries=int(app_cfg.judge.max_retries),
                retry_base=float(app_cfg.judge.retry_base),
            )
        except GeminiMissingDependencyError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("Gemini provider unavailable: google-generativeai not installed.") from exc
        return judge.batch, provider, app_cfg.judge.model or "gemini-2.5-flash"

    raise ValueError(f"Unknown judge provider '{provider}'")


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_global_seed(app_cfg.seed)

    split = cfg.get("split", "train")
    jobs_path = _resolve_root_path(_format_with_split(Path(app_cfg.judge.jobs_path), split))
    if not jobs_path.exists():
        raise FileNotFoundError(jobs_path)
    LOGGER.info("Loading judging jobs from %s", jobs_path)
    jobs = _load_jobs(jobs_path)

    batch_fn, provider_name, model_name = _build_provider(app_cfg)
    LOGGER.info("Running judge provider '%s' with model '%s' on %d jobs", provider_name, model_name, len(jobs))
    judgments = batch_fn(jobs)

    out_path = _resolve_root_path(_format_with_split(Path(app_cfg.judge.out_path), split))
    write_jsonl(out_path, [judgment.to_dict() for judgment in judgments])
    LOGGER.info("Wrote %d judgments to %s", len(judgments), out_path)

    metrics = _aggregate_metrics(judgments)
    extended_metrics: dict[str, object] = {**metrics, "provider": provider_name, "model": model_name}
    log_path = _resolve_root_path(_format_with_split(Path(app_cfg.judge.log_path), split))
    _write_metrics(log_path, split, extended_metrics)
    LOGGER.info("Judging metrics recorded at %s", log_path)
    if "avg_latency_s" in metrics:
        LOGGER.info(
            "Latency avg=%.3fs min=%.3fs max=%.3fs",
            metrics.get("avg_latency_s", 0.0),
            metrics.get("min_latency_s", 0.0),
            metrics.get("max_latency_s", 0.0),
        )


if __name__ == "__main__":
    main()
