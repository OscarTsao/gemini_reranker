"""Inference CLI for ranking models using Hydra configs."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..candidate_generation import build_judging_jobs
from ..config_schemas import AppConfig, parse_config
from ..hydra_utils import enable_speed_flags, resolve_device, set_global_seed
from ..io_utils import read_jsonl, write_jsonl
from ..mlflow_utils import get_or_create_run, log_artifact_dir, log_dataset_card
from ..models.ranker import CrossEncoderRanker
from ..schemas import Sample


LOGGER = logging.getLogger(__name__)


def _prepare_batches(
    pairs: list[dict[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        inputs = tokenizer(
            [item["criterion_text"] for item in chunk],
            [item["candidate_text"] for item in chunk],
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
    batch_size = max(1, int(cfg.get("batch_size", app_cfg.train.batch_size_per_device)))
    data_dir = Path(app_cfg.data.path)
    samples_path = data_dir / f"{split}_samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    samples = [Sample.model_validate(row) for row in read_jsonl(samples_path)]
    jobs, gen_metrics = build_judging_jobs(
        samples,
        split=split,
        k=app_cfg.candidate_gen.k,
        min_char=app_cfg.candidate_gen.min_char,
        max_char=app_cfg.candidate_gen.max_char,
        seed=app_cfg.seed,
    )
    LOGGER.info(
        "Prepared %d inference jobs (avg candidates %.2f)",
        len(jobs),
        gen_metrics.get("avg_candidates", 0.0),
    )

    tokenizer = AutoTokenizer.from_pretrained(app_cfg.model.tokenizer_name, use_fast=True)
    model = CrossEncoderRanker(app_cfg.model)
    model.to(device)
    model.eval()

    pair_inputs: list[dict[str, object]] = []
    for job in jobs:
        for idx, candidate in enumerate(job.candidates):
            pair_inputs.append(
                {
                    "job_id": job.job_id,
                    "candidate_idx": idx,
                    "criterion_text": job.criterion_text,
                    "candidate_text": candidate.text,
                }
            )

    batches = _prepare_batches(pair_inputs, tokenizer, app_cfg.data.max_length, batch_size)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    output_path = Path.cwd() / f"{split}_predictions.jsonl"

    with get_or_create_run(app_cfg, resolved_cfg):
        log_dataset_card(
            {
                "split": split,
                "num_jobs": len(jobs),
                "total_candidates": len(pair_inputs),
                "k": app_cfg.candidate_gen.k,
                "tokenizer": tokenizer.name_or_path,
            },
            f"{split}_inference",
        )
        scored: dict[str, list[tuple[int, float]]] = defaultdict(list)
        start_time = time.perf_counter()
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
                    scored[item["job_id"]].append((item["candidate_idx"], float(score)))
        elapsed = time.perf_counter() - start_time
        LOGGER.info(
            "Scored %d candidates across %d jobs in %.2fs",
            len(pair_inputs),
            len(jobs),
            elapsed,
        )

        job_lookup = {job.job_id: job for job in jobs}
        rows = []
        for job in jobs:
            candidate_scores = scored.get(job.job_id, [])
            candidate_scores.sort(key=lambda pair: pair[1], reverse=True)
            candidates_payload = []
            for idx, score in candidate_scores:
                candidate = job.candidates[idx]
                candidates_payload.append(
                    {
                        "idx": idx,
                        "text": candidate.text,
                        "model_score": score,
                        "coarse_score": candidate.score,
                        "start": candidate.start,
                        "end": candidate.end,
                        "extra": candidate.extra,
                    }
                )
            top_entry = candidates_payload[0] if candidates_payload else None
            rows.append(
                {
                    "job_id": job.job_id,
                    "note_id": job.note_id,
                    "criterion_id": job.criterion_id,
                    "criterion_text": job.criterion_text,
                    "note_text": job.note_text,
                    "candidates": candidates_payload,
                    "top1": top_entry,
                    "meta": job.meta,
                }
            )

        write_jsonl(output_path, rows)
        log_artifact_dir(output_path.parent, artifact_path="inference")
        LOGGER.info("Inference outputs written to %s", output_path)


if __name__ == "__main__":
    main()
