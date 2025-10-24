from pathlib import Path

from criteriabind.pair_builder import PairBuilderArgs, main
from criteriabind.schemas import Candidate, JudgeResult, JudgedItem
from criteriabind.io_utils import write_jsonl, read_jsonl


def test_pair_builder_creates_pairs(tmp_path: Path) -> None:
    candidates = [
        Candidate(text="Evidence one."),
        Candidate(text="Evidence two."),
    ]
    judge = JudgeResult(
        winner_index=0,
        rank=[0, 1],
        rationales="Clear winner.",
        safety={"flags": [], "notes": ""},
        rubric_version="v1",
    )
    item = JudgedItem(
        id="item1",
        note_id="note1",
        criterion="Criterion C",
        candidates=candidates,
        judge=judge,
    )
    input_path = tmp_path / "judged.jsonl"
    write_jsonl(input_path, [item])
    args = PairBuilderArgs(
        in_path=input_path,
        out_train=tmp_path / "train.jsonl",
        out_dev=tmp_path / "dev.jsonl",
        out_test=tmp_path / "test.jsonl",
        task="criteria",
        source="unittest",
        dev_ratio=0.0,
        test_ratio=0.0,
    )
    main(args)
    train_rows = list(read_jsonl(args.out_train))
    assert len(train_rows) == 1
    assert train_rows[0]["pos"] == "Evidence one."
