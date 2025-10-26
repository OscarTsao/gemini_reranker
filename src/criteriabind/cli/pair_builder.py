"""Build pairwise datasets from judged candidate outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config_schemas import parse_config
from ..io_utils import read_jsonl, write_jsonl
from ..schemas import JudgedItem


LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    split = cfg.get("split", "train")
    judgments_path = Path(cfg.get("judgments_path", Path.cwd() / f"{split}_judgments.jsonl"))
    output_path = Path(cfg.get("output_path", Path.cwd() / f"pairs_{split}.jsonl"))

    if not judgments_path.exists():
        raise FileNotFoundError(judgments_path)

    judged_items = [JudgedItem(**row) for row in read_jsonl(judgments_path)]
    LOGGER.info("Loaded %d judged items", len(judged_items))

    rows = []
    for item in judged_items:
        if not item.candidates:
            continue
        candidates = []
        for idx, candidate in enumerate(item.candidates):
            label = 1 if idx == item.judge.winner_index else 0
            candidates.append({"text": candidate.text, "label": label})
        rows.append(
            {
                "group_id": item.id,
                "note_id": item.note_id,
                "criterion": item.criterion,
                "candidates": candidates,
                "rank": item.judge.rank,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    LOGGER.info("Wrote %d pair rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
