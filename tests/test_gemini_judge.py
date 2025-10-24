"""Tests for gemini_judge module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from criteriabind.gemini_judge import (
    GeminiJudgeError,
    JudgeArgs,
    _format_prompt,
    judge_job,
)
from criteriabind.schemas import Candidate, JudgeResult, JudgedItem, JudgingJob


def test_format_prompt_variant1() -> None:
    """Test formatting prompt for variant 1."""
    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="Patient has depression",
        candidates=[
            Candidate(text="Patient is sad"),
            Candidate(text="Patient is happy"),
        ],
    )

    prompt = _format_prompt(job, 1)
    assert "Patient has depression" in prompt
    assert "[0] Patient is sad" in prompt
    assert "[1] Patient is happy" in prompt
    assert "deterministic" in prompt


def test_format_prompt_variant2() -> None:
    """Test formatting prompt for variant 2."""
    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="Test criterion",
        candidates=[Candidate(text="Test text")],
    )

    prompt = _format_prompt(job, 2)
    assert "Re-evaluate" in prompt
    assert "safer" in prompt


def test_format_prompt_variant3() -> None:
    """Test formatting prompt for variant 3."""
    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="Test criterion",
        candidates=[Candidate(text="Test text")],
    )

    prompt = _format_prompt(job, 3)
    assert "tie-breaker" in prompt


def test_judge_job_no_candidates() -> None:
    """Test judging job with no candidates."""
    from criteriabind.config import JudgeConfig

    client = MagicMock()
    job = JudgingJob(
        id="job-1", note_id="note-1", criterion="Test", candidates=[]
    )
    cfg = JudgeConfig()

    result = judge_job(client, job, cfg)
    assert result is None


def test_judge_job_safety_flags() -> None:
    """Test that safety flags cause item to be dropped."""
    from criteriabind.config import JudgeConfig

    client = MagicMock()
    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="Test",
        candidates=[Candidate(text="Test text")],
    )
    cfg = JudgeConfig()

    # Mock the API calls to return results with safety flags
    mock_result = JudgeResult(
        winner_index=0,
        rank=[0],
        rationales="Test",
        safety={"flags": ["VIOLENCE"], "notes": "Flagged"},
        rubric_version="v1",
    )

    with patch("criteriabind.gemini_judge._call_model", return_value=mock_result):
        result = judge_job(client, job, cfg)
        assert result is None  # Dropped due to safety flags


def test_judge_args_defaults() -> None:
    """Test JudgeArgs default values."""
    args = JudgeArgs(
        in_path=Path("input.jsonl"),
        out_path=Path("output.jsonl"),
    )
    assert args.model == "gemini-2.5-flash"
    assert args.rubric_version == "clinical_rubric_v1"
    assert args.drop_on_conflict is True
    assert args.mock is False


def test_judge_args_with_mock() -> None:
    """Test JudgeArgs with mock flag."""
    args = JudgeArgs(
        in_path=Path("input.jsonl"),
        out_path=Path("output.jsonl"),
        mock=True,
    )
    assert args.mock is True


def test_mock_mode_integration(tmp_path: Path) -> None:
    """Test mock mode integration."""
    # Create test input file
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="depression symptoms",
        candidates=[
            Candidate(text="Patient has depression and anxiety"),
            Candidate(text="Patient is healthy"),
        ],
    )

    with open(input_file, "w") as f:
        f.write(job.to_json() + "\n")

    # Import and run main with mock flag
    from criteriabind.gemini_judge import main

    args = JudgeArgs(in_path=input_file, out_path=output_file, mock=True)

    main(args)

    # Verify output
    assert output_file.exists()
    from criteriabind.io_utils import read_jsonl

    results = list(read_jsonl(output_file))
    assert len(results) == 1
    assert results[0]["id"] == "job-1"
    assert "judge" in results[0]
    assert results[0]["judge"]["rubric_version"] == "clinical_rubric_v1"


def test_gemini_judge_error() -> None:
    """Test GeminiJudgeError exception."""
    error = GeminiJudgeError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, RuntimeError)
