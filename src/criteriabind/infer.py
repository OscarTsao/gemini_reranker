"""Inference CLI for criteria matching and evidence binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import tyro

from .candidate_gen import generate_candidates
from .io_utils import read_jsonl, write_jsonl
from .logging_utils import get_logger
from .models import QASpanModel, load_cross_encoder
from .schemas import Candidate, Sample
from .seed import seed_everything

LOGGER = get_logger(__name__)


@dataclass
class InferArgs:
    notes_path: Path = Path("data/raw/redsm5_test.jsonl")
    output_path: Path = Path("data/proc/redsm5_inference.jsonl")
    model_name_or_path: str = "baselines/dataaug_trial_0043/model/best"
    criteria_checkpoint: Optional[Path] = None
    qa_checkpoint: Optional[Path] = None
    threshold: float = 0.5
    k: int = 5
    max_length: int = 384
    seed: int = 42


def load_samples(path: Path) -> List[Sample]:
    return [Sample(**row) for row in read_jsonl(path)]


def load_models(args: InferArgs):
    bundle = load_cross_encoder(args.model_name_or_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    if args.criteria_checkpoint and args.criteria_checkpoint.exists():
        state = torch.load(args.criteria_checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    qa_model = None
    if args.qa_checkpoint and args.qa_checkpoint.exists():
        qa_model = QASpanModel(args.model_name_or_path)
        qa_state = torch.load(args.qa_checkpoint, map_location="cpu")
        qa_model.load_state_dict(qa_state)
        qa_model.eval()
    return model, tokenizer, qa_model, bundle.max_length


def score_candidates(
    model,
    tokenizer,
    criterion: str,
    candidates: List[Candidate],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    texts1 = [criterion for _ in candidates]
    texts2 = [c.text for c in candidates]
    inputs = tokenizer(
        texts1,
        texts2,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )
    return scores.cpu().tolist()


def run_inference(args: InferArgs) -> None:
    seed_everything(args.seed)
    samples = load_samples(args.notes_path)
    LOGGER.info("Loaded %d samples", len(samples))
    model, tokenizer, qa_model, bundle_max_length = load_models(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if qa_model:
        qa_model.to(device)
        qa_model.eval()
    max_length = args.max_length
    if bundle_max_length is not None:
        max_length = min(max_length, bundle_max_length)
        LOGGER.info("Capping inference sequence length to %d", max_length)

    outputs: List[Dict] = []
    for sample in samples:
        for criterion in sample.criteria:
            candidates = generate_candidates(sample, criterion, k=args.k, span_lengths=(10, 20))
            scores = score_candidates(
                model,
                tokenizer,
                criterion,
                candidates,
                device=device,
                max_length=max_length,
            )
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top_score = ranked[0][1] if ranked else 0.0
            pred_label = int(top_score >= args.threshold)
            result = {
                "note_id": sample.id,
                "criterion": criterion,
                "pred_label": pred_label,
                "score": top_score,
                "top_evidence": [{"text": c.text, "score": s} for c, s in ranked[:3]],
            }
            if qa_model and ranked:
                inputs = tokenizer(
                    criterion,
                    sample.note_text,
                    return_offsets_mapping=True,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                offsets = inputs.pop("offset_mapping")[0].tolist()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs_qa = qa_model.model(**inputs)
                start = torch.argmax(outputs_qa.start_logits, dim=-1).item()
                end = torch.argmax(outputs_qa.end_logits, dim=-1).item()
                offset_start = offsets[start][0]
                offset_end = offsets[end][1]
                span = sample.note_text[offset_start:offset_end]
                result["qa_span"] = span
            outputs.append(result)
    write_jsonl(args.output_path, outputs)
    LOGGER.info("Wrote %d predictions to %s", len(outputs), args.output_path)


def main(args: InferArgs) -> None:
    run_inference(args)


if __name__ == "__main__":
    tyro.cli(main)
