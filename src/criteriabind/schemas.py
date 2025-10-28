"""Pydantic schemas for the reranker pipeline."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SchemaEncoder(BaseModel):
    """Base schema capable of serialising itself to JSON."""

    model_config = {"extra": "ignore"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self) -> str:
        return self.model_dump_json()


class Sample(SchemaEncoder):
    """Input sample containing a clinical note and criteria list."""

    id: str
    note_text: str
    criteria: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(SchemaEncoder):
    """Candidate span extracted from a note."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Preference(SchemaEncoder):
    """Preference comparing two candidate indices."""

    winner_idx: int
    loser_idx: int
    weight: float = 1.0


class JudgingJob(SchemaEncoder):
    """Job submitted to a judge provider for a set of candidates."""

    job_id: str
    note_id: str
    criterion_id: str
    criterion_text: str
    note_text: str
    candidates: list[Candidate]
    seed: Optional[int] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Judgment(SchemaEncoder):
    """Judging output used downstream for dataset creation and training."""

    job_id: str
    note_id: str
    criterion_id: str
    criterion_text: str
    note_text: str
    candidates: list[Candidate]
    best_idx: int
    preferences: list[Preference]
    rationale: str
    provider: str
    model: Optional[str] = None
    latency_s: Optional[float] = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Candidate",
    "JudgingJob",
    "Judgment",
    "Preference",
    "Sample",
    "SchemaEncoder",
]
