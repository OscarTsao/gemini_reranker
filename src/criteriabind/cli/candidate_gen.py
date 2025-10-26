"""Candidate generation CLI with Hydra configuration."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config_schemas import AppConfig, parse_config
from ..io_utils import read_jsonl, write_jsonl
from ..schemas import Candidate, JudgingJob, Sample
from ..text_utils import sentence_tokenize


LOGGER = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def _score_sentence(sentence_tokens: list[str], criterion_tokens: list[str]) -> float:
    overlap = len(set(sentence_tokens) & set(criterion_tokens))
    if not overlap:
        return 0.0
    return overlap / (len(criterion_tokens) + 1e-6)


def generate_candidates(note_text: str, criterion: str, k: int) -> list[Candidate]:
    if not note_text.strip():
        return []
    sentences = sentence_tokenize(note_text)
    criterion_tokens = _tokenize(criterion)
    scored = []
    for sentence in sentences:
        score = _score_sentence(_tokenize(sentence), criterion_tokens)
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_sentences = [sentence for _, sentence in scored[:k]] or sentences[:k]

    candidates: list[Candidate] = []
    seen = set()
    for sentence in top_sentences:
        if sentence in seen:
            continue
        seen.add(sentence)
        start = note_text.find(sentence)
        end = start + len(sentence) if start >= 0 else None
        candidates.append(
            Candidate(text=sentence, start=start, end=end, extra={"type": "sentence"})
        )
    return candidates


def build_judging_jobs(samples: Iterable[Sample], k: int) -> list[JudgingJob]:
    jobs: list[JudgingJob] = []
    for sample in samples:
        for criterion in sample.criteria:
            candidates = generate_candidates(sample.note_text, criterion, k)
            job_id = f"{sample.id}|{hash(criterion) & 0xFFFF:04x}"
            jobs.append(
                JudgingJob(
                    id=job_id,
                    note_id=sample.id,
                    criterion=criterion,
                    candidates=candidates,
                )
            )
    return jobs


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    split = cfg.get("split", "train")
    top_k = int(cfg.get("top_k", 5))

    data_dir = Path(app_cfg.data.path)
    samples_path = data_dir / f"{split}_samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    samples = [Sample(**row) for row in read_jsonl(samples_path)]
    jobs = build_judging_jobs(samples, top_k)
    output_path = Path.cwd() / f"{split}_jobs.jsonl"
    write_jsonl(output_path, [job.to_dict() for job in jobs])
    LOGGER.info("Wrote %d judging jobs to %s", len(jobs), output_path)


if __name__ == "__main__":
    main()
