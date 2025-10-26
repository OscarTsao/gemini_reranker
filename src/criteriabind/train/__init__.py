"""Training entrypoints and helpers."""

from .train_criteria_ranker import main as train_criteria_ranker
from .train_evidence_span import main as train_evidence_span


__all__ = ["train_criteria_ranker", "train_evidence_span"]
