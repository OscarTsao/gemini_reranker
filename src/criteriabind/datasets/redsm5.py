"""Utilities for loading the ReDSM5 dataset shipped with the repository.

The dataset is distributed as two CSV files under ``data/redsm5``:

* ``redsm5_posts.csv`` – full Reddit posts, cleaned and anonymised
* ``redsm5_annotations.csv`` – sentence-level symptom annotations with rationales

This module provides helpers that convert those CSVs into the :class:`Sample`
schema used throughout the Criteria Bind pipeline and optionally persist the
result as JSONL splits for downstream components.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..io_utils import write_jsonl
from ..logging_utils import get_logger
from ..schemas import Sample


LOGGER = get_logger(__name__)

# Canonical order used by the baseline artifacts and downstream configs.
SYMPTOM_ORDER: tuple[str, ...] = (
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
    "SPECIAL_CASE",
)

# Human-readable descriptions surfaced to Gemini and other consumers.
SYMPTOM_DESCRIPTIONS: dict[str, str] = {
    "DEPRESSED_MOOD": "Persistent depressed mood most of the day, nearly every day",
    "ANHEDONIA": "Markedly diminished interest or pleasure in almost all activities",
    "APPETITE_CHANGE": "Significant weight loss or gain, or change in appetite",
    "SLEEP_ISSUES": "Insomnia or hypersomnia nearly every day",
    "PSYCHOMOTOR": "Observable psychomotor agitation or retardation",
    "FATIGUE": "Fatigue or loss of energy nearly every day",
    "WORTHLESSNESS": "Feelings of worthlessness or excessive guilt",
    "COGNITIVE_ISSUES": "Diminished ability to think or concentrate, indecisiveness",
    "SUICIDAL_THOUGHTS": "Recurrent thoughts of death, suicidal ideation, or attempt",
    "SPECIAL_CASE": "Clinician flagged special diagnostic consideration",
}


@dataclass(slots=True)
class AnnotationRow:
    """In-memory representation of an annotation CSV row."""

    sentence_id: str
    sentence_text: str
    symptom: str
    status: int
    explanation: Optional[str]


def _load_posts(csv_path: Path) -> dict[str, str]:
    posts: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            post_id = row["post_id"].strip()
            text = row.get("text", "").strip()
            if not post_id or not text:
                continue
            posts[post_id] = text
    LOGGER.info("Loaded %d posts from %s", len(posts), csv_path)
    return posts


def _load_annotations(csv_path: Path) -> dict[str, list[AnnotationRow]]:
    grouped: dict[str, list[AnnotationRow]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            post_id = row["post_id"].strip()
            symptom = row["DSM5_symptom"].strip()
            if symptom not in SYMPTOM_DESCRIPTIONS:
                # Skip unknown symptom codes – keeps the pipeline deterministic.
                continue
            try:
                status = int(row.get("status", "0"))
            except ValueError:
                status = 0
            grouped.setdefault(post_id, []).append(
                AnnotationRow(
                    sentence_id=row["sentence_id"].strip(),
                    sentence_text=row.get("sentence_text", "").strip(),
                    symptom=symptom,
                    status=status,
                    explanation=row.get("explanation", "").strip() or None,
                )
            )
    LOGGER.info("Loaded annotations for %d posts from %s", len(grouped), csv_path)
    return grouped


def _assign_split(identifier: str, dev_ratio: float, test_ratio: float) -> str:
    """Deterministically bucket a post into train/dev/test using its identifier."""

    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + dev_ratio:
        return "dev"
    return "train"


def _build_metadata(annotations: Iterable[AnnotationRow]) -> dict[str, dict[str, object]]:
    pos_sentences: dict[str, list[str]] = {}
    neg_sentences: dict[str, list[str]] = {}
    explanations: dict[str, list[str]] = {}
    labels: dict[str, int] = {}
    for ann in annotations:
        if ann.status == 1 and ann.sentence_text:
            pos_sentences.setdefault(ann.symptom, []).append(ann.sentence_text)
        elif ann.status == 0 and ann.sentence_text:
            neg_sentences.setdefault(ann.symptom, []).append(ann.sentence_text)
        if ann.explanation:
            explanations.setdefault(ann.symptom, []).append(ann.explanation)
        labels[ann.symptom] = int(ann.status == 1 or labels.get(ann.symptom, 0))
    return {
        "symptom_labels": labels,
        "positive_sentences": pos_sentences,
        "negative_sentences": neg_sentences,
        "explanations": explanations,
    }


def load_redsm5_samples(
    data_dir: Path = Path("data/redsm5"),
    include_special_case: bool = True,
) -> list[Sample]:
    """Load the ReDSM5 dataset into :class:`Sample` objects.

    Args:
        data_dir: Directory containing the CSV files.
        include_special_case: Whether to expose the ``SPECIAL_CASE`` criterion.

    Returns:
        List of samples sorted by ``post_id``.
    """

    posts_csv = data_dir / "redsm5_posts.csv"
    annotations_csv = data_dir / "redsm5_annotations.csv"
    if not posts_csv.exists() or not annotations_csv.exists():
        raise FileNotFoundError(
            f"Expected redsm5 CSV files under {data_dir} but found none. "
            "Make sure the gated dataset is available locally.",
        )

    posts = _load_posts(posts_csv)
    annotations = _load_annotations(annotations_csv)

    samples: list[Sample] = []
    for post_id, text in posts.items():
        post_annotations = annotations.get(post_id, [])
        metadata = _build_metadata(post_annotations)

        criteria_texts: list[str] = []
        criteria_map: dict[str, dict[str, str | int]] = {}
        for symptom in SYMPTOM_ORDER:
            if symptom == "SPECIAL_CASE" and not include_special_case:
                continue
            criterion_text = SYMPTOM_DESCRIPTIONS[symptom]
            criteria_texts.append(criterion_text)
            criteria_map[criterion_text] = {
                "symptom": symptom,
                "label": metadata["symptom_labels"].get(symptom, 0),
            }

        samples.append(
            Sample(
                id=post_id,
                note_text=text,
                criteria=criteria_texts,
                metadata={
                    "symptom_metadata": metadata,
                    "criteria_map": criteria_map,
                },
            )
        )

    samples.sort(key=lambda sample: sample.id)
    LOGGER.info("Constructed %d Sample objects from ReDSM5", len(samples))
    return samples


def prepare_redsm5_splits(
    output_dir: Path,
    *,
    data_dir: Path = Path("data/redsm5"),
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    include_special_case: bool = True,
    prefix: str = "redsm5",
) -> dict[str, Path]:
    """Materialise JSONL splits from the CSV dataset.

    Args:
        output_dir: Directory where ``{split}.jsonl`` files will be written.
        data_dir: Source directory containing the CSV files.
        dev_ratio: Fraction of data assigned to the dev split.
        test_ratio: Fraction of data assigned to the test split.
        include_special_case: Whether to include the ``SPECIAL_CASE`` criterion.
        prefix: Prefix for the generated filenames.

    Returns:
        Mapping from split name to written :class:`Path`.
    """
    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1:
        raise ValueError("Invalid dev/test ratios; they must sum to less than 1 and be non-negative.")

    samples = load_redsm5_samples(data_dir=data_dir, include_special_case=include_special_case)
    buckets = {"train": [], "dev": [], "test": []}
    for sample in samples:
        split = _assign_split(sample.id, dev_ratio, test_ratio)
        buckets[split].append(sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_samples in buckets.items():
        path = output_dir / f"{prefix}_{split}.jsonl"
        write_jsonl(path, (sample.to_dict() for sample in split_samples))
        LOGGER.info("Wrote %d %s samples to %s", len(split_samples), split, path)
        paths[split] = path
    return paths
