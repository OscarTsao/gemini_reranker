from criteriabind.schemas import Candidate, JudgeResult, JudgedItem, PairwiseRow, Sample


def test_sample_serializes_roundtrip() -> None:
    sample = Sample(id="1", note_text="Example note.", criteria=["Criterion A"])
    restored = Sample.model_validate_json(sample.to_json())
    assert restored.id == "1"
    assert restored.criteria == ["Criterion A"]


def test_judged_item_structure() -> None:
    candidate = Candidate(text="Sentence")
    judge = JudgeResult(
        winner_index=0,
        rank=[0],
        rationales="Clear winner.",
        safety={"flags": [], "notes": ""},
        rubric_version="v1",
    )
    item = JudgedItem(
        id="abc",
        note_id="note",
        criterion="Criterion A",
        candidates=[candidate],
        judge=judge,
    )
    data = item.to_dict()
    assert data["judge"]["winner_index"] == 0


def test_pairwise_row_schema() -> None:
    row = PairwiseRow(
        id="p1",
        criterion="Criterion A",
        prompt="Prompt",
        pos="Positive",
        neg="Negative",
        source="unittest",
        task="criteria",
    )
    assert row.task == "criteria"
