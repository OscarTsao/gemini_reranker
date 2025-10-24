"""Candidate generation for Gemini judging."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import tyro

from .logging_utils import get_logger
from .schemas import Candidate, JudgingJob, Sample
from .io_utils import read_jsonl, write_jsonl
from .text_utils import extract_spans, sentence_tokenize

LOGGER = get_logger(__name__)


def score_sentence(criterion: str, sentence: str) -> float:
    """Score a sentence based on overlap with criterion.

    Args:
        criterion: Criterion text.
        sentence: Sentence text to score.

    Returns:
        Overlap score normalized by criterion length.
    """
    criterion_tokens = _tokenize(criterion)
    sentence_tokens = _tokenize(sentence)
    return _score_sentence(sentence_tokens, criterion_tokens)


def extract_candidates(note_text: str, criterion: str, k: int) -> List[Candidate]:
    """Extract top-k candidate sentences from note text for criterion.

    Args:
        note_text: Full clinical note text.
        criterion: Criterion to match against.
        k: Number of top candidates to return.

    Returns:
        List of Candidate objects.
    """
    if not note_text or not note_text.strip():
        return []

    sentences = sentence_tokenize(note_text)
    if not sentences:
        return []

    criterion_tokens = _tokenize(criterion)
    scored = []
    for sentence in sentences:
        score = _score_sentence(_tokenize(sentence), criterion_tokens)
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)

    candidates: List[Candidate] = []
    seen = set()
    for _, sentence in scored[:k]:
        # Deduplicate
        if sentence in seen:
            continue
        seen.add(sentence)

        start = note_text.find(sentence)
        end = start + len(sentence) if start != -1 else None
        candidates.append(
            Candidate(
                text=sentence,
                start=start,
                end=end,
                extra={"type": "sentence"},
            )
        )

    return candidates


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return [token.lower() for token in text.split() if token.strip()]


def _score_sentence(sentence_tokens: List[str], criterion_tokens: List[str]) -> float:
    """Score a sentence based on token overlap with criterion.

    Args:
        sentence_tokens: Tokens from the sentence.
        criterion_tokens: Tokens from the criterion.

    Returns:
        Overlap score normalized by criterion length.
    """
    overlap = len(set(sentence_tokens) & set(criterion_tokens))
    if not overlap:
        return 0.0
    return overlap / (len(criterion_tokens) + 1e-6)


def generate_candidates(
    sample: Sample,
    criterion: str,
    k: int,
    span_lengths: Iterable[int],
) -> List[Candidate]:
    """Generate candidate snippets from a note for a given criterion.

    Args:
        sample: Sample containing note text.
        criterion: Criterion to match against.
        k: Number of top candidates to return.
        span_lengths: Lengths of spans to extract.

    Returns:
        List of Candidate objects with text and position info.
    """
    sentences = sentence_tokenize(sample.note_text)
    criterion_tokens = _tokenize(criterion)
    scored = []
    for sentence in sentences:
        score = _score_sentence(_tokenize(sentence), criterion_tokens)
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in scored[:k]] or sentences[:k]
    candidates: List[Candidate] = []
    for sentence in selected:
        start = sample.note_text.find(sentence)
        end = start + len(sentence) if start != -1 else None
        candidates.append(
            Candidate(
                text=sentence,
                start=start,
                end=end,
                extra={"note_window": sentence, "type": "sentence"},
            )
        )
        spans = extract_spans(sample.note_text, sentence, span_lengths)
        for span_start, span_end, span_text in spans:
            candidates.append(
                Candidate(
                    text=span_text,
                    start=span_start,
                    end=span_end,
                    extra={"type": "span", "note_window": sentence},
                )
            )
    unique: List[Candidate] = []
    seen = set()
    for cand in candidates:
        key = (cand.text, cand.start, cand.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique[: max(k, len(unique))]


def build_judging_jobs(
    samples: Iterable[Sample],
    k: int,
    span_lengths: Iterable[int],
) -> List[JudgingJob]:
    """Build judging jobs for all criteria in all samples.

    Args:
        samples: Iterable of Sample objects.
        k: Number of candidates per job.
        span_lengths: Lengths of spans to extract.

    Returns:
        List of JudgingJob objects ready for judging.
    """
    jobs: List[JudgingJob] = []
    for sample in samples:
        for criterion in sample.criteria:
            job_id = hashlib.sha256(f"{sample.id}:{criterion}".encode()).hexdigest()[:16]
            candidates = generate_candidates(sample, criterion, k=k, span_lengths=span_lengths)
            job = JudgingJob(id=job_id, note_id=sample.id, criterion=criterion, candidates=candidates)
            jobs.append(job)
    return jobs


@dataclass
class CandidateGenArgs:
    """CLI arguments for candidate generation."""

    in_path: Path = Path("data/raw/train.jsonl")
    out_path: Path = Path("data/proc/judging_jobs.jsonl")
    k: int = 10
    span_lengths: tuple[int, int, int] = (10, 20, 30)


def main(args: CandidateGenArgs) -> None:
    """Main entry point for candidate generation.

    Args:
        args: Command-line arguments for candidate generation.

    Raises:
        ValueError: If k is less than 1.
        FileNotFoundError: If input file does not exist.
    """
    if args.k < 1:
        raise ValueError(f"k must be >= 1, got {args.k}")

    if not args.in_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.in_path}")

    LOGGER.info("Loading samples from %s", args.in_path)
    samples = [Sample(**row) for row in read_jsonl(args.in_path)]
    LOGGER.info("Loaded %d samples", len(samples))

    if not samples:
        LOGGER.warning("No samples loaded from input file")
        write_jsonl(args.out_path, [])
        return

    jobs = build_judging_jobs(samples, k=args.k, span_lengths=args.span_lengths)
    LOGGER.info("Writing %d judging jobs to %s", len(jobs), args.out_path)
    write_jsonl(args.out_path, jobs)
    LOGGER.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
