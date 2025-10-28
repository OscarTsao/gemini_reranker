"""Deterministic offline judge used for CI and demos."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from ..schemas import JudgingJob, Judgment, Preference


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


class MockJudge:
    """Deterministic heuristic mimicking Gemini outputs."""

    def __init__(self, *, temperature: float = 0.0, model_name: str = "offline-mock-v1") -> None:
        self.temperature = temperature
        self.model_name = model_name

    def _rank(self, job: JudgingJob) -> list[tuple[float, int]]:
        seed_source = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()
        rng = random.Random(int(seed_source[:8], 16))
        criterion_tokens = set(_tokenize(job.criterion_text))

        scored: list[tuple[float, int]] = []
        for idx, cand in enumerate(job.candidates):
            tokens = set(_tokenize(cand.text))
            overlap = len(criterion_tokens & tokens)
            coverage = overlap / max(len(tokens), 1)
            diversity = len(tokens) / max(len(criterion_tokens), 1)
            score = overlap * 2.0 + coverage + 0.2 * diversity
            if self.temperature > 0:
                score += rng.random() * self.temperature
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def score(self, job: JudgingJob) -> Judgment:
        if not job.candidates:
            raise ValueError(f"No candidates supplied for job {job.job_id}")

        ranked = self._rank(job)
        best_idx = ranked[0][1]
        preferences: list[Preference] = []
        ordered_indices = [idx for _, idx in ranked]
        for winner_rank, winner_idx in enumerate(ordered_indices):
            for loser_idx in ordered_indices[winner_rank + 1 :]:
                preferences.append(Preference(winner_idx=winner_idx, loser_idx=loser_idx))
        if not preferences and len(job.candidates) > 1:
            # Ensure at least one pair when multiple candidates exist.
            preferences.append(Preference(winner_idx=best_idx, loser_idx=ordered_indices[1]))

        return Judgment(
            job_id=job.job_id,
            note_id=job.note_id,
            criterion_id=job.criterion_id,
            criterion_text=job.criterion_text,
            note_text=job.note_text,
            candidates=job.candidates,
            best_idx=best_idx,
            preferences=preferences,
            rationale="Selected snippet with highest lexical overlap in offline mock judge.",
            provider="mock",
            model=self.model_name,
            latency_s=0.0,
            token_usage={"input_tokens": 0, "output_tokens": 0},
            meta={
                **job.meta,
                "ordering": ordered_indices,
                "scores": ranked,
            },
        )

    def batch(self, jobs: Iterable[JudgingJob]) -> list[Judgment]:
        return [self.score(job) for job in jobs]
