"""Build pairwise/listwise datasets from judged items."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Literal, cast

import tyro

from .io_utils import read_jsonl, write_jsonl
from .logging_utils import get_logger
from .schemas import JudgedItem, PairwiseRow

LOGGER = get_logger(__name__)


def _assign_split(note_id: str, dev_ratio: float, test_ratio: float) -> str:
    """Assign a note to train/dev/test split deterministically.

    Args:
        note_id: Unique identifier for the note.
        dev_ratio: Proportion of data for dev set (0-1).
        test_ratio: Proportion of data for test set (0-1).

    Returns:
        Split name: "train", "dev", or "test".
    """
    digest = hashlib.sha256(note_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + dev_ratio:
        return "dev"
    return "train"


@dataclass
class PairBuilderArgs:
    """CLI arguments for pair building."""

    in_path: Path
    out_train: Path
    out_dev: Path
    out_test: Path
    task: str = "criteria"
    source: str = "gemini_judge:v1"
    dev_ratio: float = 0.1
    test_ratio: float = 0.1


def main(args: PairBuilderArgs) -> None:
    """Main entry point for pair building.

    Args:
        args: Command-line arguments for pair building.

    Raises:
        ValueError: If task is not 'criteria' or 'evidence', or ratios invalid.
        FileNotFoundError: If input file does not exist.
    """
    if args.task not in {"criteria", "evidence"}:
        raise ValueError("task must be either 'criteria' or 'evidence'")

    if not 0.0 <= args.dev_ratio <= 1.0:
        raise ValueError(f"dev_ratio must be in [0, 1], got {args.dev_ratio}")

    if not 0.0 <= args.test_ratio <= 1.0:
        raise ValueError(f"test_ratio must be in [0, 1], got {args.test_ratio}")

    if args.dev_ratio + args.test_ratio >= 1.0:
        raise ValueError(
            f"dev_ratio + test_ratio must be < 1.0, got {args.dev_ratio + args.test_ratio}"
        )

    if not args.in_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.in_path}")
    judged = [JudgedItem(**row) for row in read_jsonl(args.in_path)]
    LOGGER.info("Loaded %d judged items", len(judged))
    splits = {"train": [], "dev": [], "test": []}
    for item in judged:
        split = _assign_split(item.note_id, args.dev_ratio, args.test_ratio)
        winner_idx = item.judge.winner_index
        if winner_idx < 0 or winner_idx >= len(item.candidates):
            continue
        positive = item.candidates[winner_idx].text
        positive_window = item.candidates[winner_idx].extra.get("note_window", positive)
        for idx, candidate in enumerate(item.candidates):
            if idx == winner_idx:
                continue
            pair_id = f"{item.id}-{winner_idx}-{idx}"
            prompt = f"{item.criterion}\n\n{positive_window}"
            row = PairwiseRow(
                id=pair_id,
                criterion=item.criterion,
                prompt=prompt,
                pos=positive,
                neg=candidate.text,
                source=args.source,
                task=cast("Literal['criteria','evidence']", args.task),
            )
            splits[split].append(row)
    LOGGER.info(
        "Generated %d train / %d dev / %d test pairs",
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
    )
    write_jsonl(args.out_train, splits["train"])
    write_jsonl(args.out_dev, splits["dev"])
    write_jsonl(args.out_test, splits["test"])
    LOGGER.info("Saved pairwise datasets.")


if __name__ == "__main__":
    tyro.cli(main)
