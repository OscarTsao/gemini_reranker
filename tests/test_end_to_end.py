"""End-to-end integration tests for the full pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from criteriabind.candidate_gen import CandidateGenArgs, main as candidate_gen_main
from criteriabind.gemini_judge import JudgeArgs, main as judge_main
from criteriabind.io_utils import read_jsonl
from criteriabind.pair_builder import PairBuilderArgs, main as pair_builder_main
from criteriabind.schemas import JudgedItem, JudgingJob, PairwiseRow, Sample


@pytest.fixture
def sample_data(tmp_path: Path) -> Path:
    """Create sample input data."""
    input_file = tmp_path / "input.jsonl"

    samples = [
        Sample(
            id="note-1",
            note_text=(
                "Patient reports persistent depressed mood for three weeks. "
                "Significant weight loss noted. "
                "Sleep disturbance present. "
                "Patient denies suicidal ideation."
            ),
            criteria=[
                "Depressed mood most of the day",
                "Significant weight change",
                "Sleep disturbance",
            ],
        ),
        Sample(
            id="note-2",
            note_text=(
                "Patient exhibits anxiety in social situations. "
                "Avoids public speaking. "
                "Heart rate increases in crowds. "
                "No panic attacks reported."
            ),
            criteria=[
                "Anxiety in social situations",
                "Avoidance behavior",
            ],
        ),
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")

    return input_file


def test_candidate_generation_pipeline(sample_data: Path, tmp_path: Path) -> None:
    """Test candidate generation step."""
    output_file = tmp_path / "jobs.jsonl"

    args = CandidateGenArgs(in_path=sample_data, out_path=output_file, k=5)
    candidate_gen_main(args)

    assert output_file.exists()

    jobs = list(read_jsonl(output_file))
    assert len(jobs) > 0

    # Verify schema
    job = JudgingJob(**jobs[0])
    assert job.id
    assert job.note_id
    assert job.criterion
    assert len(job.candidates) > 0


def test_mock_judging_pipeline(tmp_path: Path) -> None:
    """Test Gemini judging with mock mode."""
    # Create judging jobs
    jobs_file = tmp_path / "jobs.jsonl"
    judged_file = tmp_path / "judged.jsonl"

    job = JudgingJob(
        id="job-1",
        note_id="note-1",
        criterion="depressed mood",
        candidates=[
            {"text": "Patient has depressed mood and low energy"},
            {"text": "Patient is cheerful"},
            {"text": "Blood pressure normal"},
        ],
    )

    with open(jobs_file, "w", encoding="utf-8") as f:
        f.write(job.to_json() + "\n")

    # Run mock judging
    args = JudgeArgs(in_path=jobs_file, out_path=judged_file, mock=True)
    judge_main(args)

    assert judged_file.exists()

    judged_items = list(read_jsonl(judged_file))
    assert len(judged_items) == 1

    # Verify schema
    item = JudgedItem(**judged_items[0])
    assert item.judge.winner_index >= 0
    assert len(item.judge.rank) == len(item.candidates)
    assert item.judge.rubric_version == "clinical_rubric_v1"


def test_pair_building_pipeline(tmp_path: Path) -> None:
    """Test pair building step."""
    judged_file = tmp_path / "judged.jsonl"
    train_file = tmp_path / "train.jsonl"
    dev_file = tmp_path / "dev.jsonl"
    test_file = tmp_path / "test.jsonl"

    # Create judged items
    judged_items = []
    for i in range(10):
        item = JudgedItem(
            id=f"job-{i}",
            note_id=f"note-{i}",
            criterion=f"criterion-{i}",
            candidates=[
                {"text": f"candidate-{i}-0"},
                {"text": f"candidate-{i}-1"},
                {"text": f"candidate-{i}-2"},
            ],
            judge={
                "winner_index": 0,
                "rank": [0, 1, 2],
                "rationales": "Test rationale",
                "safety": {"flags": [], "notes": "Safe"},
                "rubric_version": "v1",
            },
        )
        judged_items.append(item)

    with open(judged_file, "w", encoding="utf-8") as f:
        for item in judged_items:
            f.write(item.to_json() + "\n")

    # Run pair builder
    args = PairBuilderArgs(
        in_path=judged_file,
        out_train=train_file,
        out_dev=dev_file,
        out_test=test_file,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    pair_builder_main(args)

    assert train_file.exists()
    assert dev_file.exists()
    assert test_file.exists()

    # Verify pairs
    train_pairs = list(read_jsonl(train_file))
    assert len(train_pairs) > 0

    pair = PairwiseRow(**train_pairs[0])
    assert pair.criterion
    assert pair.pos
    assert pair.neg
    assert pair.task in ["criteria", "evidence"]


def test_full_pipeline_integration(sample_data: Path, tmp_path: Path) -> None:
    """Test full pipeline: candidate_gen -> judge (mock) -> pair_builder."""
    jobs_file = tmp_path / "jobs.jsonl"
    judged_file = tmp_path / "judged.jsonl"
    train_file = tmp_path / "train.jsonl"
    dev_file = tmp_path / "dev.jsonl"
    test_file = tmp_path / "test.jsonl"

    # Step 1: Candidate generation
    candidate_gen_main(CandidateGenArgs(in_path=sample_data, out_path=jobs_file, k=5))
    assert jobs_file.exists()

    jobs = list(read_jsonl(jobs_file))
    assert len(jobs) > 0

    # Step 2: Mock judging
    judge_main(JudgeArgs(in_path=jobs_file, out_path=judged_file, mock=True))
    assert judged_file.exists()

    judged_items = list(read_jsonl(judged_file))
    assert len(judged_items) > 0

    # Step 3: Pair building
    pair_builder_main(
        PairBuilderArgs(
            in_path=judged_file,
            out_train=train_file,
            out_dev=dev_file,
            out_test=test_file,
        )
    )

    assert train_file.exists()
    train_pairs = list(read_jsonl(train_file))
    assert len(train_pairs) > 0

    # Verify final output schema
    pair = PairwiseRow(**train_pairs[0])
    assert pair.id
    assert pair.criterion
    assert pair.pos
    assert pair.neg


def test_pipeline_determinism(sample_data: Path, tmp_path: Path) -> None:
    """Test that pipeline is deterministic with same seed."""
    # Run 1
    jobs_file_1 = tmp_path / "jobs_1.jsonl"
    candidate_gen_main(
        CandidateGenArgs(in_path=sample_data, out_path=jobs_file_1, k=5)
    )

    # Run 2
    jobs_file_2 = tmp_path / "jobs_2.jsonl"
    candidate_gen_main(
        CandidateGenArgs(in_path=sample_data, out_path=jobs_file_2, k=5)
    )

    # Compare outputs
    with open(jobs_file_1, "r", encoding="utf-8") as f:
        content_1 = f.read()
    with open(jobs_file_2, "r", encoding="utf-8") as f:
        content_2 = f.read()

    assert content_1 == content_2


def test_pipeline_with_empty_candidates(tmp_path: Path) -> None:
    """Test pipeline handles empty candidate lists gracefully."""
    jobs_file = tmp_path / "jobs_empty.jsonl"
    judged_file = tmp_path / "judged_empty.jsonl"

    # Create job with empty candidates
    job = JudgingJob(
        id="job-empty",
        note_id="note-1",
        criterion="test",
        candidates=[],
    )

    with open(jobs_file, "w", encoding="utf-8") as f:
        f.write(job.to_json() + "\n")

    # Should handle gracefully (skip the job)
    judge_main(JudgeArgs(in_path=jobs_file, out_path=judged_file, mock=True))

    judged_items = list(read_jsonl(judged_file))
    # Empty candidates should be skipped
    assert len(judged_items) == 0
