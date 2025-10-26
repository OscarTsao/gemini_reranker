"""Judge CLI supporting mock and Gemini backends."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config_schemas import AppConfig, parse_config
from ..io_utils import read_jsonl, write_jsonl
from ..judge.mock_judge import MockJudge
from ..schemas import JudgingJob


LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    split = cfg.get("split", "train")
    jobs_path = Path(cfg.get("jobs_path", Path.cwd() / f"{split}_jobs.jsonl"))
    output_path = Path(cfg.get("output_path", Path.cwd() / f"{split}_judgments.jsonl"))

    if not jobs_path.exists():
        raise FileNotFoundError(jobs_path)

    jobs = [JudgingJob(**row) for row in read_jsonl(jobs_path)]
    LOGGER.info("Loaded %d jobs from %s", len(jobs), jobs_path)

    provider = app_cfg.judge.provider
    if provider == "mock":
        judge = MockJudge(temperature=float(app_cfg.judge.temperature))
        judged = judge.batch(jobs)
    else:  # pragma: no cover - real Gemini path
        raise NotImplementedError("Gemini provider not implemented in offline demo")

    write_jsonl(output_path, [item.to_dict() for item in judged])
    LOGGER.info("Wrote %d judged items to %s", len(judged), output_path)


if __name__ == "__main__":
    main()
