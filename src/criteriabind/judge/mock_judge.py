"""Deterministic offline judge used for CI and demos."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from ..schemas import JudgedItem, JudgeResult, JudgingJob


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


class MockJudge:
    """Deterministic heuristic mimicking Gemini outputs."""

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature

    def score(self, job: JudgingJob) -> JudgedItem:
        if not job.candidates:
            raise ValueError(f"No candidates supplied for job {job.id}")

        seed_source = hashlib.sha256(job.id.encode("utf-8")).hexdigest()
        rng = random.Random(int(seed_source[:8], 16))
        criterion_tokens = set(_tokenize(job.criterion))

        ranked: list[tuple[float, int]] = []
        for idx, cand in enumerate(job.candidates):
            tokens = set(_tokenize(cand.text))
            overlap = len(criterion_tokens & tokens)
            coverage = overlap / max(len(tokens), 1)
            diversity = len(tokens) / max(len(criterion_tokens), 1)
            jitter = rng.random() * self.temperature
            score = overlap * 2.0 + coverage + 0.2 * diversity + jitter
            ranked.append((score, idx))

        ranked.sort(key=lambda item: item[0], reverse=True)
        winner_index = ranked[0][1]
        ordering = [idx for _, idx in ranked]

        judge_result = JudgeResult(
            winner_index=winner_index,
            rank=ordering,
            rationales="Selected snippet with highest lexical overlap in mock judge.",
            safety={"flags": [], "notes": "offline-mock"},
            rubric_version="mock_v1",
        )
        return JudgedItem(
            id=job.id,
            note_id=job.note_id,
            criterion=job.criterion,
            candidates=job.candidates,
            judge=judge_result,
        )

    def batch(self, jobs: Iterable[JudgingJob]) -> list[JudgedItem]:
        return [self.score(job) for job in jobs]
