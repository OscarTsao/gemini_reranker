"""Collation helpers and sampling utilities for DataLoaders."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase


class EmptyFeatureError(ValueError):
    """Raised when a collate function receives no features."""

    def __init__(self) -> None:
        super().__init__("features must be non-empty")


def _pad_feature_sequences(
    features: Sequence[dict[str, torch.Tensor]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Pad variable-length transformer inputs into a batch tensor."""

    if not features:
        raise EmptyFeatureError
    keys = features[0].keys()
    padded: dict[str, torch.Tensor] = {}
    for key in keys:
        sequences = [feature[key] for feature in features]
        pad_value = tokenizer.pad_token_id or 0
        if key != "input_ids":
            pad_value = 0
        padded[key] = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    return padded


def make_ranker_collate(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[Sequence[dict[str, object]]], dict[str, object]]:
    """Factory for the ranker collate_fn with dynamic padding."""

    def collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
        pos_features = [item["pos"] for item in batch]
        neg_features = [item["neg"] for item in batch]
        weights = torch.tensor([float(item.get("weight", 1.0)) for item in batch], dtype=torch.float32)
        return {
            "pair_ids": [item.get("pair_id") for item in batch],
            "group_ids": [item["group_id"] for item in batch],
            "note_ids": [item["note_id"] for item in batch],
            "criteria": [item["criterion"] for item in batch],
            "pos_text": [item["pos_text"] for item in batch],
            "neg_text": [item["neg_text"] for item in batch],
            "pos_inputs": _pad_feature_sequences(pos_features, tokenizer),
            "neg_inputs": _pad_feature_sequences(neg_features, tokenizer),
            "weights": weights,
            "meta": [item.get("meta", {}) for item in batch],
        }

    return collate


def make_span_collate(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[Sequence[dict[str, object]]], dict[str, object]]:
    """Factory for span extraction collation."""

    def collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
        encodings = [item["encoding"] for item in batch]
        inputs = _pad_feature_sequences(encodings, tokenizer)
        start_positions = torch.tensor([item["start_position"] for item in batch], dtype=torch.long)
        end_positions = torch.tensor([item["end_position"] for item in batch], dtype=torch.long)
        return {
            "uids": [item["uid"] for item in batch],
            "note_ids": [item["note_id"] for item in batch],
            "criteria": [item["criterion"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "answers": [item["answers"] for item in batch],
            "offsets": [item["offsets"] for item in batch],
            "inputs": inputs,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

    return collate


class BucketBatchSampler(Sampler[list[int]]):
    """Length-aware sampler to reduce padding by forming local buckets."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size: int | None = None,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = bucket_size or batch_size * 4
        self.seed = seed
        self._iteration = 0

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        if n == 0:
            return iter([])
        indices = list(range(n))
        indices.sort(key=lambda idx: self.lengths[idx])

        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        for start in range(0, n, self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            if self.shuffle:
                rng.shuffle(bucket)
            for sub_start in range(0, len(bucket), self.batch_size):
                batch = bucket[sub_start : sub_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)
