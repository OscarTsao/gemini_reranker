"""Evaluation utilities for ranking and span extraction models."""

from __future__ import annotations

import logging
from collections import defaultdict

import torch


LOGGER = logging.getLogger(__name__)


class Evaluator:
    """Task-specific evaluation harness with early-stopping support."""

    def __init__(self, task: str, metric: str) -> None:
        self.task = task
        self.metric = metric

    def evaluate(self, model, dataloader, device: str, amp_dtype: torch.dtype) -> dict[str, float]:
        if self.task == "criteria_ranker":
            return self._evaluate_ranker(model, dataloader, device, amp_dtype)
        if self.task == "evidence_span":
            return self._evaluate_span(model, dataloader, device, amp_dtype)
        raise ValueError(f"Unsupported task {self.task}")

    def better(self, current: float, best: float) -> bool:
        return current > best

    def _evaluate_ranker(self, model, dataloader, device: str, amp_dtype: torch.dtype) -> dict[str, float]:
        model.eval()
        total_pairs = 0
        correct_pairs = 0
        group_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pos": [], "neg": []})
        with torch.no_grad():
            for batch in dataloader:
                pos_inputs = {k: v.to(device) for k, v in batch["pos_inputs"].items()}
                neg_inputs = {k: v.to(device) for k, v in batch["neg_inputs"].items()}
                context = torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype, enabled=device == "cuda" and amp_dtype != torch.float32)
                with context:
                    pos_scores = model(pos_inputs)
                    neg_scores = model(neg_inputs)
                correct_pairs += (pos_scores > neg_scores).sum().item()
                total_pairs += pos_scores.numel()
                for idx, group_id in enumerate(batch["group_ids"]):
                    group_scores[group_id]["pos"].append(float(pos_scores[idx].detach().cpu()))
                    group_scores[group_id]["neg"].append(float(neg_scores[idx].detach().cpu()))

        map_scores = []
        for scores in group_scores.values():
            combined = [(score, 1) for score in scores["pos"]] + [(score, 0) for score in scores["neg"]]
            if not combined:
                continue
            combined.sort(key=lambda item: item[0], reverse=True)
            hits = 0
            precision_sum = 0.0
            for rank, (_, label) in enumerate(combined, start=1):
                if label == 1:
                    hits += 1
                    precision_sum += hits / rank
            map_scores.append(precision_sum / max(hits, 1))

        pair_acc = correct_pairs / max(total_pairs, 1)
        map_at_10 = sum(map_scores) / max(len(map_scores), 1)
        model.train()
        return {
            "pair_accuracy": pair_acc,
            "map@10": map_at_10,
            "groups_evaluated": float(len(group_scores)),
        }

    def _evaluate_span(self, model, dataloader, device: str, amp_dtype: torch.dtype) -> dict[str, float]:
        model.eval()
        matches = 0
        total = 0
        iou_scores = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
                context = torch.autocast(
                    device_type="cuda" if device == "cuda" else "cpu",
                    dtype=amp_dtype,
                    enabled=device == "cuda" and amp_dtype != torch.float32,
                )
                with context:
                    start_logits, end_logits = model(inputs)
                start_positions = torch.argmax(start_logits, dim=-1)
                end_positions = torch.argmax(end_logits, dim=-1)
                for idx in range(start_positions.size(0)):
                    answers = batch["answers"][idx]
                    if not answers:
                        continue
                    gold = answers[0]
                    pred_span = _token_to_char_span(
                        start_positions[idx].item(),
                        end_positions[idx].item(),
                        [tuple(pair) for pair in batch["offsets"][idx]],
                    )
                    gold_span = (gold["start"], gold["end"])
                    iou = _span_iou(pred_span, gold_span)
                    matches += 1 if iou >= 0.5 else 0
                    total += 1
                    iou_scores.append(iou)
        model.train()
        avg_iou = sum(iou_scores) / max(len(iou_scores), 1)
        f1 = matches / max(total, 1)
        return {
            "f1_at_iou": f1,
            "avg_iou": avg_iou,
        }


def _token_to_char_span(start_idx: int, end_idx: int, offsets: list[tuple[int, int]]) -> tuple[int, int]:
    start_idx = max(0, min(start_idx, len(offsets) - 1))
    end_idx = max(0, min(end_idx, len(offsets) - 1))
    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]
    if end_char < start_char:
        end_char = start_char
    return start_char, end_char


def _span_iou(pred: tuple[int, int], gold: tuple[int, int]) -> float:
    pred_start, pred_end = pred
    gold_start, gold_end = gold
    intersection = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    if union <= 0:
        return 0.0
    return intersection / union
