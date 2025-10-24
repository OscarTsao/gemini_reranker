"""Tests for candidate_gen module."""

from __future__ import annotations

from pathlib import Path

import pytest

from criteriabind.candidate_gen import (
    CandidateGenArgs,
    extract_candidates,
    score_sentence,
)
from criteriabind.schemas import Candidate, JudgingJob, Sample


def test_score_sentence_basic() -> None:
    """Test basic sentence scoring."""
    criterion = "patient has depression"
    sentence = "the patient exhibits symptoms of depression"

    score = score_sentence(criterion, sentence)
    assert score > 0.0
    assert isinstance(score, float)


def test_score_sentence_no_overlap() -> None:
    """Test sentence scoring with no keyword overlap."""
    criterion = "patient has diabetes"
    sentence = "blood pressure is normal"

    score = score_sentence(criterion, sentence)
    # Should still return a score (may be low)
    assert isinstance(score, float)


def test_extract_candidates_basic() -> None:
    """Test basic candidate extraction."""
    note_text = (
        "Patient reports feeling sad. "
        "They have lost interest in activities. "
        "Sleep is disturbed. "
        "Appetite is decreased."
    )
    criterion = "depressed mood"

    candidates = extract_candidates(note_text, criterion, k=3)

    assert len(candidates) <= 3
    assert all(isinstance(c, Candidate) for c in candidates)
    assert all(c.text for c in candidates)


def test_extract_candidates_k_larger_than_sentences() -> None:
    """Test extraction when k is larger than number of sentences."""
    note_text = "Short note. Only two sentences."
    criterion = "test"

    candidates = extract_candidates(note_text, criterion, k=10)

    # Should return all available sentences
    assert len(candidates) <= 2


def test_extract_candidates_empty_note() -> None:
    """Test extraction from empty note."""
    note_text = ""
    criterion = "test"

    candidates = extract_candidates(note_text, criterion, k=5)

    assert len(candidates) == 0


def test_extract_candidates_whitespace_only() -> None:
    """Test extraction from whitespace-only note."""
    note_text = "   \n\n   \t  "
    criterion = "test"

    candidates = extract_candidates(note_text, criterion, k=5)

    assert len(candidates) == 0


def test_candidate_gen_args_defaults() -> None:
    """Test CandidateGenArgs default values."""
    args = CandidateGenArgs(
        in_path=Path("input.jsonl"),
        out_path=Path("output.jsonl"),
    )
    assert args.k == 10  # Default is 10, not 8


def test_candidate_gen_args_custom_k() -> None:
    """Test CandidateGenArgs with custom k."""
    args = CandidateGenArgs(
        in_path=Path("input.jsonl"),
        out_path=Path("output.jsonl"),
        k=5,
    )
    assert args.k == 5


def test_extract_candidates_deduplication() -> None:
    """Test that duplicate sentences are handled properly."""
    note_text = (
        "Patient is sad. "
        "Patient is sad. "  # Duplicate
        "Patient is very sad. "
        "Blood pressure normal."
    )
    criterion = "sad"

    candidates = extract_candidates(note_text, criterion, k=5)

    # Should have some deduplication logic
    assert len(candidates) <= 3


def test_candidate_gen_integration(tmp_path: Path) -> None:
    """Test full candidate generation pipeline."""
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    # Create test sample
    sample = Sample(
        id="test-1",
        note_text=(
            "Patient has depression. "
            "Patient is anxious. "
            "Patient sleeps poorly. "
            "Blood pressure is normal."
        ),
        criteria=["depression", "anxiety"],
    )

    with open(input_file, "w") as f:
        f.write(sample.to_json() + "\n")

    # Run candidate generation
    from criteriabind.candidate_gen import main

    args = CandidateGenArgs(in_path=input_file, out_path=output_file, k=3)
    main(args)

    # Verify output
    assert output_file.exists()

    from criteriabind.io_utils import read_jsonl

    jobs = list(read_jsonl(output_file))
    assert len(jobs) == 2  # One job per criterion

    job = JudgingJob(**jobs[0])
    assert job.note_id == "test-1"
    assert len(job.candidates) <= 3
