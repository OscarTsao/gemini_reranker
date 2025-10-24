"""Pydantic schemas used across the project."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SchemaEncoder(BaseModel):
    """Base schema capable of serialising itself to JSON."""

    model_config = {"extra": "ignore"}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self) -> str:
        return self.model_dump_json()


class Sample(SchemaEncoder):
    """Input sample containing a clinical note and criteria list."""

    id: str
    note_text: str
    criteria: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Candidate(SchemaEncoder):
    """Candidate sentence/span extracted from a note."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class JudgeResult(SchemaEncoder):
    """Result returned from Gemini judging."""

    winner_index: int
    rank: List[int]
    rationales: str
    safety: Dict[str, Any]
    rubric_version: str


class JudgedItem(SchemaEncoder):
    """Stored judging output for downstream training."""

    id: str
    note_id: str
    criterion: str
    candidates: List[Candidate]
    judge: JudgeResult


class PairwiseRow(SchemaEncoder):
    """Pairwise training row with positive/negative candidates."""

    id: str
    criterion: str
    prompt: str
    pos: str
    neg: str
    source: str
    task: Literal["criteria", "evidence"]


class JudgingJob(SchemaEncoder):
    """Job submitted to Gemini judge for a set of candidates."""

    id: str
    note_id: str
    criterion: str
    candidates: List[Candidate]


__all__ = [
    "Candidate",
    "JudgeResult",
    "JudgedItem",
    "JudgingJob",
    "PairwiseRow",
    "Sample",
    "SchemaEncoder",
]
