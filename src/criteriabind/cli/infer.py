"""Inference CLI for ranking models using Hydra configs."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config_schemas import AppConfig, parse_config
from ..hydra_utils import enable_speed_flags, resolve_device, set_global_seed
from ..io_utils import read_jsonl, write_jsonl
from ..mlflow_utils import get_or_create_run, log_artifact_dir, log_dataset_card
from ..models.ranker import CrossEncoderRanker


LOGGER = logging.getLogger(__name__)


def _prepare_batches(
    pairs: list[dict[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    stride = 32
    for start in range(0, len(pairs), stride):
        chunk = pairs[start : start + stride]
        inputs = tokenizer(
            [item["criterion"] for item in chunk],
            [item["cand_text"] for item in chunk],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        batches.append({**inputs, "meta": chunk})
    return batches


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_global_seed(app_cfg.seed)
    device, amp_dtype = resolve_device(app_cfg)
    enable_speed_flags(app_cfg)

    split = cfg.get("split", "test")
    data_dir = Path(app_cfg.data.path)
    pairs_path = data_dir / f"pairs_{split}.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(pairs_path)

    tokenizer = AutoTokenizer.from_pretrained(app_cfg.model.tokenizer_name, use_fast=True)

    model = CrossEncoderRanker(app_cfg.model)
    model.to(device)
    model.eval()

    raw_pairs: list[dict[str, object]] = []
    group_counts: dict[str, int] = defaultdict(int)
    for row in read_jsonl(pairs_path):
        for candidate in row.get("candidates", []):
            raw_pairs.append(
                {
                    "group_id": row["group_id"],
                    "note_id": row.get("note_id", ""),
                    "criterion": row["criterion"],
                    "cand_text": candidate["text"],
                }
            )
            group_counts[row["group_id"]] += 1

    batches = _prepare_batches(raw_pairs, tokenizer, app_cfg.data.max_length)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    output_path = Path.cwd() / f"{split}_predictions.jsonl"

    with get_or_create_run(app_cfg, resolved_cfg):
        log_dataset_card(
            {
                "split": split,
                "num_pairs": len(raw_pairs),
                "num_groups": len(group_counts),
                "tokenizer": tokenizer.name_or_path,
            },
            f"{split}_pairs",
        )
        scored: dict[str, list[dict[str, object]]] = defaultdict(list)
        with torch.no_grad():
            for batch in batches:
                meta = batch.pop("meta")
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device_type="cuda" if device == "cuda" else "cpu",
                    dtype=amp_dtype,
                    enabled=device == "cuda" and amp_dtype != torch.float32,
                ):
                    logits = model(inputs)
                scores = logits.detach().cpu().tolist()
                for item, score in zip(meta, scores, strict=False):
                    scored[item["group_id"]].append(
                        {
                            "criterion": item["criterion"],
                            "candidate": item["cand_text"],
                            "score": float(score),
                            "note_id": item["note_id"],
                        }
                    )

        rows = []
        for group_id, candidates in scored.items():
            candidates.sort(key=lambda entry: entry["score"], reverse=True)
            rows.append({"group_id": group_id, "candidates": candidates})

        write_jsonl(output_path, rows)
        log_artifact_dir(output_path.parent, artifact_path="inference")
        LOGGER.info("Inference outputs written to %s", output_path)


if __name__ == "__main__":
    main()
