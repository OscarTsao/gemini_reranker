"""Candidate generation CLI driven by Hydra configs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from ..candidate_generation import build_judging_jobs
from ..config_schemas import AppConfig, parse_config
from ..hydra_utils import set_global_seed
from ..io_utils import read_jsonl, write_jsonl
from ..schemas import Sample


LOGGER = logging.getLogger(__name__)


def _resolve_root_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        base = Path(get_original_cwd())
    except Exception:  # pragma: no cover - hydra not initialised in some contexts
        base = Path.cwd()
    return (base / path).resolve()


def _format_with_split(path: Path, split: str) -> Path:
    return Path(str(path).replace("{split}", split))

def _write_metrics_csv(path: Path, split: str, metrics: dict[str, float]) -> None:
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
        row.update({key: metrics[key] for key in metrics})
        writer.writerow(row)


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_global_seed(app_cfg.seed)

    split = cfg.get("split", "train")
    data_dir = _resolve_root_path(Path(app_cfg.data.path))
    samples_path = data_dir / f"{split}_samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    samples = [Sample.model_validate(row) for row in read_jsonl(samples_path)]
    jobs, metrics = build_judging_jobs(
        samples,
        split=split,
        k=app_cfg.candidate_gen.k,
        min_char=app_cfg.candidate_gen.min_char,
        max_char=app_cfg.candidate_gen.max_char,
        seed=app_cfg.seed,
    )

    jobs_path = _format_with_split(Path(app_cfg.candidate_gen.jobs_path), split)
    jobs_path = _resolve_root_path(jobs_path)
    write_jsonl(jobs_path, [job.to_dict() for job in jobs])
    LOGGER.info("Wrote %d judging jobs to %s", len(jobs), jobs_path)

    if metrics:
        recall = metrics.get("recall_at_k")
        if recall is not None:
            LOGGER.info(
                "Recall@%d: %.3f (%d/%d)",
                app_cfg.candidate_gen.k,
                recall,
                int(metrics.get("recall_hits", 0)),
                int(metrics.get("recall_total", 0)),
            )
        LOGGER.info(
            "Average candidates per job: %.2f",
            metrics.get("avg_candidates", 0.0),
        )

    metrics_path_config = app_cfg.candidate_gen.metrics_path
    if metrics_path_config:
        metrics_path = _format_with_split(Path(metrics_path_config), split)
        metrics_path = _resolve_root_path(metrics_path)
        _write_metrics_csv(metrics_path, split, metrics)
        LOGGER.info("Candidate generation metrics appended to %s", metrics_path)


if __name__ == "__main__":
    main()
