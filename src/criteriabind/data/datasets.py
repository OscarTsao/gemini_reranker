"""Dataset builders with tokenization caching and bucketed batching."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config_schemas import AppConfig
from ..io_utils import read_jsonl


LOGGER = logging.getLogger(__name__)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        from hydra.utils import get_original_cwd

        root = Path(get_original_cwd())
    except Exception:
        root = Path.cwd()
    return (root / path).resolve()


def _fingerprint(
    files: Sequence[Path],
    tokenizer_name: str,
    max_length: int,
    extra: str,
) -> str:
    """Build a deterministic fingerprint covering files and tokenization params."""

    hasher = hashlib.sha256()
    hasher.update(tokenizer_name.encode("utf-8"))
    hasher.update(str(max_length).encode("utf-8"))
    hasher.update(extra.encode("utf-8"))
    for path in sorted(files):
        hasher.update(path.as_posix().encode("utf-8"))
        if path.exists():
            stat = path.stat()
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
    return hasher.hexdigest()[:16]


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class PairRecord:
    group_id: str
    note_id: str
    criterion: str
    positive: dict[str, object]
    negative: dict[str, object]


class RankingDataset(Dataset):
    """Pairwise ranking dataset with cached tokenization."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        tokenizer: PreTrainedTokenizerBase,
        cfg: AppConfig,
        split: str,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split = split
        self.cache_path = self._build_cache_path()
        self.features = self._load_or_process()
        self.group_ids = [feat["group_id"] for feat in self.features]
        self.lengths = [max(feat["pos_length"], feat["neg_length"]) for feat in self.features]

    def _build_cache_path(self) -> Path:
        cache_dir = _resolve_path(Path(self.cfg.data.cache_dir))
        _ensure_cache_dir(cache_dir)
        data_dir = Path(self.cfg.data.path)
        files = [data_dir / f"pairs_{self.split}.jsonl"]
        fingerprint = _fingerprint(
            files,
            self.tokenizer.name_or_path,
            self.cfg.data.max_length,
            f"{len(self.records)}_{self.cfg.data.padding}_{self.cfg.data.truncation}",
        )
        return cache_dir / f"ranker_{self.split}_{fingerprint}.pt"

    def _load_or_process(self) -> list[dict[str, object]]:
        if self.cache_path.exists():
            LOGGER.info("Loading cached ranker features from %s", self.cache_path)
            return torch.load(self.cache_path)
        LOGGER.info("Tokenizing %d ranking pairs for %s split", len(self.records), self.split)
        features: list[dict[str, object]] = []
        max_length = self.cfg.data.max_length
        truncation = bool(self.cfg.data.truncation)
        for record in self.records:
            pos_inputs = self.tokenizer(
                record.criterion,
                record.positive["text"],
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
                padding=False,
            )
            neg_inputs = self.tokenizer(
                record.criterion,
                record.negative["text"],
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
                padding=False,
            )
            pos = {k: v.squeeze(0) for k, v in pos_inputs.items()}
            neg = {k: v.squeeze(0) for k, v in neg_inputs.items()}
            features.append(
                {
                    "group_id": record.group_id,
                    "note_id": record.note_id,
                    "criterion": record.criterion,
                    "pos_text": record.positive["text"],
                    "neg_text": record.negative["text"],
                    "pos": pos,
                    "neg": neg,
                    "pos_length": int(pos.get("attention_mask", torch.tensor([])).sum().item())
                    if "attention_mask" in pos
                    else int(pos["input_ids"].numel()),
                    "neg_length": int(neg.get("attention_mask", torch.tensor([])).sum().item())
                    if "attention_mask" in neg
                    else int(neg["input_ids"].numel()),
                }
            )
        torch.save(features, self.cache_path)
        return features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.features[index]


@dataclass
class SpanRecord:
    uid: str
    note_id: str
    criterion: str
    context: str
    answers: list[dict[str, int | str]]


class SpanDataset(Dataset):
    """Span extraction dataset with token-level supervision."""

    def __init__(
        self,
        records: Sequence[SpanRecord],
        tokenizer: PreTrainedTokenizerBase,
        cfg: AppConfig,
        split: str,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split = split
        self.cache_path = self._build_cache_path()
        self.features = self._load_or_process()

    def _build_cache_path(self) -> Path:
        cache_dir = Path(self.cfg.data.cache_dir)
        _ensure_cache_dir(cache_dir)
        data_dir = Path(self.cfg.data.path)
        files = [data_dir / f"spans_{self.split}.jsonl"]
        fingerprint = _fingerprint(
            files,
            self.tokenizer.name_or_path,
            self.cfg.data.max_length,
            f"{len(self.records)}_{self.cfg.data.padding}_{self.cfg.data.truncation}",
        )
        return cache_dir / f"spans_{self.split}_{fingerprint}.pt"

    def _load_or_process(self) -> list[dict[str, object]]:
        if self.cache_path.exists():
            LOGGER.info("Loading cached span features from %s", self.cache_path)
            return torch.load(self.cache_path)

        LOGGER.info("Tokenizing %d span samples for %s split", len(self.records), self.split)
        features: list[dict[str, object]] = []
        max_length = self.cfg.data.max_length
        truncation = bool(self.cfg.data.truncation)

        for record in self.records:
            encoding = self.tokenizer(
                record.criterion,
                record.context,
                truncation=truncation,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=False,
            )
            offsets_tensor = encoding.pop("offset_mapping").squeeze(0)
            start_position, end_position = _find_span_positions(offsets_tensor, record.answers)
            feature = {
                "uid": record.uid,
                "note_id": record.note_id,
                "criterion": record.criterion,
                "context": record.context,
                "answers": record.answers,
                "encoding": {k: v.squeeze(0) for k, v in encoding.items()},
                "start_position": start_position,
                "end_position": end_position,
                "offsets": [(int(start), int(end)) for start, end in offsets_tensor.tolist()],
            }
            features.append(feature)

        torch.save(features, self.cache_path)
        return features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.features[index]


def _find_span_positions(
    offsets: torch.Tensor,
    answers: list[dict[str, int | str]],
) -> tuple[int, int]:
    """Project character-level answers to token offsets."""

    if not answers:
        return 0, 0

    char_start = answers[0]["start"]
    char_end = answers[0]["end"]
    start_position, end_position = 0, 0
    for idx, (start, end) in enumerate(offsets.tolist()):
        if start <= char_start < end:
            start_position = idx
        if start < char_end <= end:
            end_position = idx
            break
    return start_position, end_position


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    tokenizer: PreTrainedTokenizerBase
    card: dict[str, object]


def build_ranker_datasets(cfg: AppConfig) -> DatasetBundle:
    """Construct train/validation pairwise datasets."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_records = list(_iter_pair_records(cfg, split="train"))
    val_records = list(_iter_pair_records(cfg, split="val"))
    train_dataset = RankingDataset(train_records, tokenizer, cfg, split="train")
    val_dataset = RankingDataset(val_records, tokenizer, cfg, split="val")
    dataset_card = {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "max_length": cfg.data.max_length,
        "tokenizer": tokenizer.name_or_path,
    }
    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        tokenizer=tokenizer,
        card=dataset_card,
    )


def load_ranker_dataset(
    cfg: AppConfig,
    split: str,
) -> tuple[RankingDataset, PreTrainedTokenizerBase, dict[str, object]]:
    """Load a single split of the ranking dataset."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    records = list(_iter_pair_records(cfg, split))
    dataset = RankingDataset(records, tokenizer, cfg, split=split)
    dataset_card = {
        "records": len(records),
        "split": split,
        "max_length": cfg.data.max_length,
        "tokenizer": tokenizer.name_or_path,
    }
    return dataset, tokenizer, dataset_card


def build_span_datasets(cfg: AppConfig) -> DatasetBundle:
    """Construct train/validation span datasets."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_records = list(_iter_span_records(cfg, split="train"))
    val_records = list(_iter_span_records(cfg, split="val"))
    train_dataset = SpanDataset(train_records, tokenizer, cfg, split="train")
    val_dataset = SpanDataset(val_records, tokenizer, cfg, split="val")
    dataset_card = {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "max_length": cfg.data.max_length,
        "tokenizer": tokenizer.name_or_path,
    }
    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        tokenizer=tokenizer,
        card=dataset_card,
    )


def _iter_pair_records(cfg: AppConfig, split: str) -> Iterator[PairRecord]:
    data_dir = _resolve_path(Path(cfg.data.path))
    path = data_dir / f"pairs_{split}.jsonl"
    if not path.exists():
        LOGGER.warning("Pair dataset split %s missing at %s", split, path)
        return
    count = 0
    for row in read_jsonl(path):
        candidates = row.get("candidates", [])
        positives = [cand for cand in candidates if cand.get("label", 0) == 1]
        negatives = [cand for cand in candidates if cand.get("label", 0) == 0]
        if not positives or not negatives:
            continue
        for pos in positives:
            for neg in negatives:
                yield PairRecord(
                    group_id=row["group_id"],
                    note_id=row.get("note_id", row["group_id"]),
                    criterion=row["criterion"],
                    positive=pos,
                    negative=neg,
                )
                count += 1
                if cfg.data.max_samples and count >= cfg.data.max_samples:
                    return


def _iter_span_records(cfg: AppConfig, split: str) -> Iterator[SpanRecord]:
    data_dir = _resolve_path(Path(cfg.data.path))
    path = data_dir / f"spans_{split}.jsonl"
    if not path.exists():
        LOGGER.warning("Span dataset split %s missing at %s", split, path)
        return
    count = 0
    for row in read_jsonl(path):
        answers = row.get("answers", [])
        if not answers:
            continue
        yield SpanRecord(
            uid=row.get("uid", row["note_id"]),
            note_id=row["note_id"],
            criterion=row["criterion"],
            context=row["context"],
            answers=answers,
        )
        count += 1
        if cfg.data.max_samples and count >= cfg.data.max_samples:
            return
