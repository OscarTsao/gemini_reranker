"""Build pairwise and listwise datasets from judged outputs."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from ..config_schemas import AppConfig, parse_config
from ..hydra_utils import set_global_seed
from ..io_utils import read_jsonl, write_jsonl
from ..schemas import Candidate, Judgment, Preference


LOGGER = logging.getLogger(__name__)


def _resolve_root_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        base = Path(get_original_cwd())
    except Exception:  # pragma: no cover
        base = Path.cwd()
    return (base / path).resolve()


def _format_with_split(path: Path, split: str) -> Path:
    return Path(str(path).replace("{split}", split))


def _candidate_payload(candidate: Candidate, idx: int) -> dict[str, object]:
    return {
        "idx": idx,
        "text": candidate.text,
        "start": candidate.start,
        "end": candidate.end,
        "score": candidate.score,
        "extra": candidate.extra,
    }


def _pair_id(judgment: Judgment, pref: Preference) -> str:
    return f"{judgment.job_id}|{pref.winner_idx}-{pref.loser_idx}"


def _build_pairs(
    judgment: Judgment,
    *,
    neg_sampling: str,
    top_m: int | None,
    margin_weight: bool,
) -> list[dict[str, object]]:
    winners_seen: defaultdict[int, int] = defaultdict(int)
    seen_pairs: set[tuple[int, int]] = set()
    pairs: list[dict[str, object]] = []

    for pref in judgment.preferences:
        pair_key = (pref.winner_idx, pref.loser_idx)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        if neg_sampling == "topM":
            limit = top_m or 1
            if winners_seen[pref.winner_idx] >= limit:
                continue
        winners_seen[pref.winner_idx] += 1

        try:
            winner = judgment.candidates[pref.winner_idx]
            loser = judgment.candidates[pref.loser_idx]
        except IndexError:
            LOGGER.warning(
                "Preference indices out of range for job %s (winner=%s loser=%s)",
                judgment.job_id,
                pref.winner_idx,
                pref.loser_idx,
            )
            continue

        weight = float(pref.weight or 1.0)
        if margin_weight and winner.score is not None and loser.score is not None:
            delta = abs(float(winner.score) - float(loser.score))
            weight *= max(0.1, delta)

        pairs.append(
            {
                "pair_id": _pair_id(judgment, pref),
                "job_id": judgment.job_id,
                "note_id": judgment.note_id,
                "criterion_id": judgment.criterion_id,
                "criterion_text": judgment.criterion_text,
                "note_text": judgment.note_text,
                "winner": _candidate_payload(winner, pref.winner_idx),
                "loser": _candidate_payload(loser, pref.loser_idx),
                "weight": weight,
                "meta": {
                    "preference_weight": pref.weight,
                },
            }
        )

    return pairs


def _build_ranking(judgment: Judgment) -> list[dict[str, object]]:
    stats = {
        idx: {"wins": 0, "losses": 0}
        for idx in range(len(judgment.candidates))
    }
    for pref in judgment.preferences:
        if pref.winner_idx in stats:
            stats[pref.winner_idx]["wins"] += 1
        if pref.loser_idx in stats:
            stats[pref.loser_idx]["losses"] += 1
    ordering = sorted(
        stats.items(),
        key=lambda item: (item[1]["wins"], -item[1]["losses"], item[0]),
        reverse=True,
    )
    ranking = []
    for order, (idx, counts) in enumerate(ordering):
        ranking.append(
            {
                "idx": idx,
                "wins": counts["wins"],
                "losses": counts["losses"],
                "order": order,
                "score": judgment.candidates[idx].score,
            }
        )
    return ranking


def _sync_to_data_dir(
    source: Path,
    data_dir: Path,
    template: str,
    split: str,
) -> Path:
    target = data_dir / template.format(split=split)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_global_seed(app_cfg.seed)

    split = cfg.get("split", "train")
    judgments_path = _resolve_root_path(_format_with_split(Path(app_cfg.pair_builder.judgments_path), split))
    if not judgments_path.exists():
        raise FileNotFoundError(judgments_path)
    judgments = [Judgment.model_validate(row) for row in read_jsonl(judgments_path)]
    LOGGER.info("Loaded %d judgments from %s", len(judgments), judgments_path)

    pairwise_rows: list[dict[str, object]] = []
    if app_cfg.pair_builder.mode in {"pairwise", "both"}:
        for judgment in judgments:
            pairs = _build_pairs(
                judgment,
                neg_sampling=app_cfg.pair_builder.neg_sampling,
                top_m=app_cfg.pair_builder.top_m,
                margin_weight=bool(app_cfg.pair_builder.weight.margin),
            )
            pairwise_rows.extend(pairs)
        LOGGER.info("Built %d pairwise rows", len(pairwise_rows))

    listwise_rows: list[dict[str, object]] = []
    if app_cfg.pair_builder.mode in {"listwise", "both"}:
        for judgment in judgments:
            listwise_rows.append(
                {
                    "job_id": judgment.job_id,
                    "note_id": judgment.note_id,
                    "criterion_id": judgment.criterion_id,
                    "criterion_text": judgment.criterion_text,
                    "note_text": judgment.note_text,
                    "best_idx": judgment.best_idx,
                    "ranking": _build_ranking(judgment),
                    "candidates": [
                        _candidate_payload(candidate, idx)
                        for idx, candidate in enumerate(judgment.candidates)
                    ],
                    "meta": judgment.meta,
                }
            )
        LOGGER.info("Built %d listwise rows", len(listwise_rows))

    data_dir = _resolve_root_path(Path(app_cfg.data.path))
    outputs: dict[str, str] = {}

    if pairwise_rows:
        pairwise_path = _resolve_root_path(_format_with_split(Path(app_cfg.pair_builder.pairwise_path), split))
        write_jsonl(pairwise_path, pairwise_rows)
        LOGGER.info("Pairwise dataset written to %s", pairwise_path)
        outputs["pairwise_path"] = pairwise_path.as_posix()
        synced = _sync_to_data_dir(
            pairwise_path,
            data_dir,
            app_cfg.data.pairwise_filename,
            split,
        )
        outputs["dataset_pairwise_path"] = synced.as_posix()
        LOGGER.info("Pairwise dataset synced to %s", synced)

    if listwise_rows:
        listwise_path = _resolve_root_path(_format_with_split(Path(app_cfg.pair_builder.listwise_path), split))
        write_jsonl(listwise_path, listwise_rows)
        LOGGER.info("Listwise dataset written to %s", listwise_path)
        outputs["listwise_path"] = listwise_path.as_posix()
        listwise_template = app_cfg.data.listwise_filename
        synced = _sync_to_data_dir(
            listwise_path,
            data_dir,
            listwise_template,
            split,
        )
        outputs["dataset_listwise_path"] = synced.as_posix()
        LOGGER.info("Listwise dataset synced to %s", synced)

    metadata = {
        "split": split,
        "num_judgments": len(judgments),
        "num_pairwise": len(pairwise_rows),
        "num_listwise": len(listwise_rows),
        "neg_sampling": app_cfg.pair_builder.neg_sampling,
        "top_m": app_cfg.pair_builder.top_m,
        "margin_weight": bool(app_cfg.pair_builder.weight.margin),
        **outputs,
    }
    metadata_path = _resolve_root_path(_format_with_split(Path(app_cfg.pair_builder.metadata_path), split))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Pair builder metadata written to %s", metadata_path)


if __name__ == "__main__":
    main()
