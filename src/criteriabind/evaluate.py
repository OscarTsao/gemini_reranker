"""Evaluation CLI for criteria and evidence tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tyro

from .io_utils import read_jsonl
from .logging_utils import get_logger
from .metrics import classification_metrics, hit_at_k, qa_em_f1

LOGGER = get_logger(__name__)


@dataclass
class CriteriaEvalArgs:
    predictions: Path
    threshold: float = 0.5


@dataclass
class EvidenceEvalArgs:
    predictions: Path


@dataclass
class EvalArgs:
    task: str
    predictions: Path
    threshold: float = 0.5


def evaluate_criteria(predictions_path: Path, threshold: float) -> None:
    records = list(read_jsonl(predictions_path))
    y_true = [int(row["label"]) for row in records]
    y_scores = [float(row["score"]) for row in records]
    metrics = classification_metrics(y_true, y_scores, threshold)
    for key, value in metrics.items():
        LOGGER.info("%s: %.4f", key, value)


def evaluate_evidence(predictions_path: Path) -> None:
    records = list(read_jsonl(predictions_path))
    preds = [row["pred_text"] for row in records]
    refs = [row["answer_text"] for row in records]
    metrics = qa_em_f1(preds, refs)
    for key, value in metrics.items():
        LOGGER.info("%s: %.4f", key, value)
    ranked = [row.get("top_candidates", []) for row in records]
    gold_ids = [row.get("gold_id", "") for row in records]
    if all(ranked) and all(gold_ids):
        hit = hit_at_k(ranked, gold_ids, k=3)
        LOGGER.info("Hit@3: %.4f", hit)


def main(args: EvalArgs) -> None:
    if args.task == "criteria":
        evaluate_criteria(args.predictions, args.threshold)
    elif args.task == "evidence":
        evaluate_evidence(args.predictions)
    else:
        raise ValueError("Unknown task")


if __name__ == "__main__":
    tyro.cli(main)
