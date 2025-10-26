"""Generate a lightweight redsm5-style dataset for demo and CI runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..io_utils import write_jsonl


@dataclass
class CriterionExample:
    text: str
    label: int
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)


@dataclass
class NoteExample:
    note_id: str
    note_text: str
    criteria: list[CriterionExample]


TRAIN_NOTES: list[NoteExample] = [
    NoteExample(
        note_id="train_001",
        note_text=(
            "The patient reports feeling hopeless for the last three months. "
            "She barely enjoys painting anymore despite it being her favorite hobby. "
            "Her appetite is unchanged, and she still prepares balanced meals. "
            "She typically sleeps only four hours each night with frequent awakenings. "
            "Despite tiredness, she jogs twice a week with her neighbor."
        ),
        criteria=[
            CriterionExample(
                text="Persistent depressed mood most of the day, nearly every day",
                label=1,
                positives=["The patient reports feeling hopeless for the last three months."],
                negatives=["Despite tiredness, she jogs twice a week with her neighbor."],
            ),
            CriterionExample(
                text="Markedly diminished interest or pleasure in almost all activities",
                label=1,
                positives=[
                    "She barely enjoys painting anymore despite it being her favorite hobby.",
                ],
                negatives=[
                    "She typically sleeps only four hours each night with frequent awakenings.",
                ],
            ),
            CriterionExample(
                text="Significant weight loss or gain, or change in appetite",
                label=0,
                positives=[],
                negatives=[
                    "Her appetite is unchanged, and she still prepares balanced meals.",
                ],
            ),
        ],
    ),
    NoteExample(
        note_id="train_002",
        note_text=(
            "He wakes up rested and attends college classes daily. "
            "He denies hopelessness and says his mood is steady. "
            "Assignments are completed on time, and he studies with friends. "
            "He does mention reduced appetite after recovering from the flu."
        ),
        criteria=[
            CriterionExample(
                text="Persistent depressed mood most of the day, nearly every day",
                label=0,
                positives=[],
                negatives=["He denies hopelessness and says his mood is steady."],
            ),
            CriterionExample(
                text="Significant weight loss or gain, or change in appetite",
                label=1,
                positives=[
                    "He does mention reduced appetite after recovering from the flu.",
                ],
                negatives=[
                    "Assignments are completed on time, and he studies with friends.",
                ],
            ),
        ],
    ),
    NoteExample(
        note_id="train_003",
        note_text=(
            "Sleep averages three hours during the workweek due to racing thoughts. "
            "On weekends he naps repeatedly but never feels rested. "
            "He enjoys hiking, which temporarily improves his energy. "
            "He feels guilty about missing family dinners."
        ),
        criteria=[
            CriterionExample(
                text="Insomnia or hypersomnia nearly every day",
                label=1,
                positives=[
                    "Sleep averages three hours during the workweek due to racing thoughts.",
                    "On weekends he naps repeatedly but never feels rested.",
                ],
                negatives=["He enjoys hiking, which temporarily improves his energy."],
            ),
            CriterionExample(
                text="Feelings of worthlessness or excessive guilt",
                label=1,
                positives=["He feels guilty about missing family dinners."],
                negatives=["He enjoys hiking, which temporarily improves his energy."],
            ),
        ],
    ),
]

VAL_NOTES: list[NoteExample] = [
    NoteExample(
        note_id="val_001",
        note_text=(
            "Client describes persistent sadness though she attends support groups weekly. "
            "She reports normal appetite and says cooking meals with roommates is enjoyable. "
            "Falling asleep takes over an hour most nights, and she wakes at 3 a.m. without reason."
        ),
        criteria=[
            CriterionExample(
                text="Persistent depressed mood most of the day, nearly every day",
                label=1,
                positives=[
                    "Client describes persistent sadness though she attends support groups weekly.",
                ],
                negatives=[
                    "She reports normal appetite and enjoys cooking with roommates.",
                ],
            ),
            CriterionExample(
                text="Insomnia or hypersomnia nearly every day",
                label=1,
                positives=[
                    "Falling asleep takes over an hour most nights, and she wakes at 3 a.m.",
                ],
                negatives=[
                    "She reports normal appetite and enjoys cooking with roommates.",
                ],
            ),
        ],
    )
]

TEST_NOTES: list[NoteExample] = [
    NoteExample(
        note_id="test_001",
        note_text=(
            "Participant shares that he feels energetic after starting morning walks. "
            "He denies any change in appetite. "
            "He sleeps seven hours per night and rarely wakes up."
        ),
        criteria=[
            CriterionExample(
                text="Significant weight loss or gain, or change in appetite",
                label=0,
                positives=[],
                negatives=["He denies any change in appetite."],
            ),
            CriterionExample(
                text="Insomnia or hypersomnia nearly every day",
                label=0,
                positives=[],
                negatives=["He sleeps seven hours per night and rarely wakes up."],
            ),
        ],
    )
]


def _serialize_samples(notes: Iterable[NoteExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for note in notes:
        criteria_texts = [criterion.text for criterion in note.criteria]
        criteria_details = {
            criterion.text: {
                "label": criterion.label,
                "positives": criterion.positives,
                "negatives": criterion.negatives,
            }
            for criterion in note.criteria
        }
        rows.append(
            {
                "id": note.note_id,
                "note_text": note.note_text,
                "criteria": criteria_texts,
                "metadata": {"criteria_details": criteria_details},
            }
        )
    return rows


def _build_pairs(notes: Iterable[NoteExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for note in notes:
        sentences = [
            sentence.strip()
            for sentence in note.note_text.split(". ")
            if sentence.strip()
        ]
        for criterion in note.criteria:
            positives = list(criterion.positives)
            negatives = list(criterion.negatives) or [
                sent for sent in sentences if sent not in positives
            ]
            if not positives or not negatives:
                continue
            candidate_list = [{"text": text, "label": 1} for text in positives]
            candidate_list.extend({"text": text, "label": 0} for text in negatives[:3])
            rows.append(
                {
                    "group_id": f"{note.note_id}|{criterion.text}",
                    "note_id": note.note_id,
                    "criterion": criterion.text,
                    "candidates": candidate_list,
                }
            )
    return rows


def _build_spans(notes: Iterable[NoteExample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for note in notes:
        for criterion in note.criteria:
            for positive in criterion.positives:
                if positive not in note.note_text:
                    continue
                start = note.note_text.index(positive)
                end = start + len(positive)
                rows.append(
                    {
                        "uid": f"{note.note_id}|{criterion.text}|{start}",
                        "note_id": note.note_id,
                        "criterion": criterion.text,
                        "context": note.note_text,
                        "answers": [{"text": positive, "start": start, "end": end}],
                    }
                )
    return rows


def write_split(
    root: Path,
    split: str,
    notes: Iterable[NoteExample],
) -> None:
    notes_list = list(notes)
    write_jsonl(root / f"{split}_samples.jsonl", _serialize_samples(notes_list))
    write_jsonl(root / f"pairs_{split}.jsonl", _build_pairs(notes_list))
    write_jsonl(root / f"spans_{split}.jsonl", _build_spans(notes_list))


def main() -> None:
    root = Path("demo_data/redsm5")
    root.mkdir(parents=True, exist_ok=True)
    write_split(root, "train", TRAIN_NOTES)
    write_split(root, "val", VAL_NOTES)
    write_split(root, "test", TEST_NOTES)


if __name__ == "__main__":
    main()
