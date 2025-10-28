"""Reusable candidate generation utilities."""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from .schemas import Candidate, JudgingJob, Sample
from .text_utils import sentence_tokenize


def criterion_id(note_id: str, criterion_text: str) -> str:
    """Stable identifier for a note/criterion pair."""

    digest = hashlib.sha256(f"{note_id}|{criterion_text}".encode("utf-8")).hexdigest()[:12]
    return f"{note_id}-{digest}"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def _lexical_overlap(a: str, b: str) -> float:
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / (len(tokens_b) + 1e-6)


def _candidate_pool(note_text: str, criterion_text: str) -> list[dict[str, object]]:
    sentences = [sentence.strip() for sentence in sentence_tokenize(note_text) if sentence.strip()]
    pool: list[dict[str, object]] = []
    for idx, sentence in enumerate(sentences):
        start = note_text.find(sentence)
        end = start + len(sentence) if start >= 0 else None
        score = _lexical_overlap(sentence, criterion_text)
        pool.append(
            {
                "text": sentence,
                "start": start if start >= 0 else None,
                "end": end,
                "length": len(sentence),
                "score": score,
                "source": "sentence",
                "sentence_index": idx,
            }
        )
    if not pool:
        pool.append(
            {
                "text": note_text,
                "start": 0,
                "end": len(note_text),
                "length": len(note_text),
                "score": _lexical_overlap(note_text, criterion_text),
                "source": "note",
                "sentence_index": 0,
            }
        )
    return pool


def select_candidates(
    note_text: str,
    criterion_text: str,
    *,
    k: int,
    min_char: Optional[int],
    max_char: Optional[int],
) -> list[Candidate]:
    """Select top-K candidate spans using lexical overlap heuristics."""

    pool = _candidate_pool(note_text, criterion_text)
    filtered = [
        cand
        for cand in pool
        if (min_char is None or cand["length"] >= min_char)
        and (max_char is None or cand["length"] <= max_char)
    ]
    ranked = sorted(filtered or pool, key=lambda item: (item["score"], -item["length"]), reverse=True)
    seen: set[str] = set()
    chosen: list[dict[str, object]] = []
    for cand in ranked:
        text = cand["text"]
        if text in seen:
            continue
        seen.add(text)
        chosen.append(cand)
        if len(chosen) >= k:
            break
    if len(chosen) < min(k, len(pool)):
        for cand in pool:
            text = cand["text"]
            if text in seen:
                continue
            chosen.append(cand)
            seen.add(text)
            if len(chosen) >= min(k, len(pool)):
                break
    return [
        Candidate(
            text=cand["text"],
            start=cand.get("start"),
            end=cand.get("end"),
            score=float(cand.get("score", 0.0)),
            extra={
                "source": cand.get("source", "sentence"),
                "length": cand.get("length"),
                "sentence_index": cand.get("sentence_index"),
            },
        )
        for cand in chosen
    ]


def _positive_recall(candidates: list[Candidate], positives: Iterable[str]) -> bool:
    if not candidates:
        return False
    positive_list = [pos.strip().lower() for pos in positives if pos.strip()]
    if not positive_list:
        return False
    candidate_texts = [cand.text.lower() for cand in candidates]
    return any(any(pos in cand for cand in candidate_texts) for pos in positive_list)


def build_judging_jobs(
    samples: Iterable[Sample],
    *,
    split: str,
    k: int,
    min_char: Optional[int],
    max_char: Optional[int],
    seed: int,
) -> tuple[list[JudgingJob], dict[str, float]]:
    jobs: list[JudgingJob] = []
    total_with_gold = 0
    hits = 0

    for sample in samples:
        criteria_meta = sample.metadata.get("criteria_details", {})
        for criterion_text in sample.criteria:
            candidates = select_candidates(
                sample.note_text,
                criterion_text,
                k=k,
                min_char=min_char,
                max_char=max_char,
            )
            criterion_meta = criteria_meta.get(criterion_text, {})
            positives = criterion_meta.get("positives", [])
            has_recall = _positive_recall(candidates, positives)
            if positives:
                total_with_gold += 1
                hits += int(has_recall)
            criterion_hash = criterion_id(sample.id, criterion_text)
            job_id = f"{sample.id}|{criterion_hash}"
            jobs.append(
                JudgingJob(
                    job_id=job_id,
                    note_id=sample.id,
                    criterion_id=criterion_hash,
                    criterion_text=criterion_text,
                    note_text=sample.note_text,
                    candidates=candidates,
                    seed=seed,
                    meta={
                        "split": split,
                        "candidate_gen": {
                            "k": k,
                            "min_char": min_char,
                            "max_char": max_char,
                            "method": "sentence_lexical_overlap",
                        },
                        "positives": positives,
                        "coarse_scores": [
                            {
                                "score": cand.score,
                                "length": cand.extra.get("length"),
                                "sentence_index": cand.extra.get("sentence_index"),
                            }
                            for cand in candidates
                        ],
                    },
                )
            )

    metrics: dict[str, float] = {}
    if total_with_gold:
        metrics["recall_at_k"] = hits / total_with_gold
        metrics["recall_hits"] = hits
        metrics["recall_total"] = total_with_gold
    metrics["num_jobs"] = float(len(jobs))
    metrics["avg_candidates"] = (
        sum(len(job.candidates) for job in jobs) / len(jobs) if jobs else 0.0
    )
    metrics["k"] = float(k)
    return jobs, metrics


__all__ = ["criterion_id", "select_candidates", "build_judging_jobs"]
