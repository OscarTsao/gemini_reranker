"""Training loop for evidence binding QA model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import tyro

from .config import build_run_config, TrainingConfig
from .io_utils import read_jsonl
from .logging_utils import get_logger
from .mlflow_utils import mlflow_run
from .models import QASpanModel
from .seed import seed_everything

LOGGER = get_logger(__name__)


@dataclass
class EvidenceRow:
    """Training example for evidence binding."""

    note_id: str
    criterion: str
    context: str
    answer_text: str
    answer_start: int
    answer_end: int
    neg_start: Optional[int] = None
    neg_end: Optional[int] = None


class EvidenceDataset(Dataset):
    """Dataset for QA training."""

    def __init__(self, rows: Sequence[EvidenceRow], tokenizer, max_length: int) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.criterion,
            row.context,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0]
        start_pos, end_pos = _char_to_token_indices(offsets, row.answer_start, row.answer_end)
        neg_positions = None
        if row.neg_start is not None and row.neg_end is not None:
            neg_positions = _char_to_token_indices(offsets, row.neg_start, row.neg_end)
        features = {key: value.squeeze(0) for key, value in encoded.items()}
        features["start_positions"] = torch.tensor(start_pos)
        features["end_positions"] = torch.tensor(end_pos)
        features["neg_start_positions"] = (
            torch.tensor(neg_positions[0], dtype=torch.long)
            if neg_positions
            else torch.tensor(-1, dtype=torch.long)
        )
        features["neg_end_positions"] = (
            torch.tensor(neg_positions[1], dtype=torch.long)
            if neg_positions
            else torch.tensor(-1, dtype=torch.long)
        )
        return features


def _char_to_token_indices(offsets, answer_start: int, answer_end: int) -> Tuple[int, int]:
    start_token = 0
    end_token = 0
    for idx, (start, end) in enumerate(offsets.tolist()):
        if start <= answer_start < end:
            start_token = idx
        if start < answer_end <= end:
            end_token = idx
            break
    if start_token > end_token:
        end_token = start_token
    return start_token, end_token


def load_rows(path: Path) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for record in read_jsonl(path):
        rows.append(
            EvidenceRow(
                note_id=record["note_id"],
                criterion=record["criterion"],
                context=record["context"],
                answer_text=record["answer_text"],
                answer_start=record["answer_start"],
                answer_end=record["answer_end"],
                neg_start=record.get("neg_start"),
                neg_end=record.get("neg_end"),
            )
        )
    return rows


@dataclass
class TrainArgs:
    config: Optional[Path] = None
    train_path: Path = Path("data/pairs/evidence_train.jsonl")
    dev_path: Optional[Path] = Path("data/pairs/evidence_dev.jsonl")
    output_dir: Path = Path("data/models/evidence")
    model_name_or_path: str = "bert-base-uncased"
    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 42
    mixed_precision: str = "bf16"
    margin: float = 0.3
    max_length: int = 512
    mlflow_run_name: Optional[str] = None


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


def evaluate(model: QASpanModel, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    em_scores: List[float] = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if "positions" not in k}
            outputs = model.model(**inputs)
            start_preds = torch.argmax(outputs.start_logits, dim=-1)
            end_preds = torch.argmax(outputs.end_logits, dim=-1)
            gold_start = batch["start_positions"]
            gold_end = batch["end_positions"]
            matches = (start_preds.cpu() == gold_start) & (end_preds.cpu() == gold_end)
            em_scores.extend(matches.float().tolist())
    model.train()
    return sum(em_scores) / len(em_scores) if em_scores else 0.0


def train(args: TrainArgs, cfg: Optional[TrainingConfig] = None) -> None:
    seed_everything(args.seed if cfg is None else cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path if cfg is None else cfg.model_name_or_path)
    model = QASpanModel(args.model_name_or_path if cfg is None else cfg.model_name_or_path, margin=args.margin if cfg is None else cfg.margin).to(device)

    train_rows = load_rows(args.train_path)
    dev_rows = load_rows(args.dev_path) if args.dev_path and args.dev_path.exists() else []

    train_dataset = EvidenceDataset(train_rows, tokenizer, args.max_length if cfg is None else cfg.max_length)
    dev_dataset = EvidenceDataset(dev_rows, tokenizer, args.max_length if cfg is None else cfg.max_length) if dev_rows else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if cfg is None else cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = (
        DataLoader(
            dev_dataset,
            batch_size=args.batch_size if cfg is None else cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if dev_dataset
        else None
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr if cfg is None else cfg.optimizer.lr,
        weight_decay=args.weight_decay if cfg is None else cfg.optimizer.weight_decay,
    )
    optimizer.zero_grad()
    total_steps = max(1, len(train_loader) * (args.epochs if cfg is None else cfg.epochs))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps if cfg is None else cfg.scheduler.warmup_steps,
        num_training_steps=total_steps,
    )
    precision = (args.mixed_precision if cfg is None else cfg.mixed_precision).lower()
    use_autocast = precision in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    accumulation = args.grad_accum_steps if cfg is None else cfg.grad_accum_steps
    total_epochs = args.epochs if cfg is None else cfg.epochs
    output_dir = args.output_dir if cfg is None else Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run(args.mlflow_run_name if cfg is None else cfg.mlflow_run_name, params={"epochs": total_epochs, "lr": optimizer.param_groups[0]["lr"]}):
        for epoch in range(total_epochs):
            for step, batch in enumerate(train_loader):
                inputs = {k: v.to(device) for k, v in batch.items() if "neg" not in k and "positions" not in k}
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                neg_start = batch["neg_start_positions"].to(device)
                neg_end = batch["neg_end_positions"].to(device)
                autocast_dtype = torch.bfloat16 if precision == "bf16" else None
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    model_kwargs = dict(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )
                    if "token_type_ids" in inputs:
                        model_kwargs["token_type_ids"] = inputs["token_type_ids"]
                    neg_mask = (neg_start >= 0) & (neg_end >= 0)
                    if neg_mask.any():
                        model_kwargs["neg_start_positions"] = torch.where(
                            neg_mask, neg_start, torch.zeros_like(neg_start)
                        )
                        model_kwargs["neg_end_positions"] = torch.where(
                            neg_mask, neg_end, torch.zeros_like(neg_end)
                        )
                        model_kwargs["neg_mask"] = neg_mask
                    outputs = model(**model_kwargs)
                    loss = outputs["loss"] / accumulation
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % accumulation == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            if dev_loader:
                metric = evaluate(model, dev_loader, device)
                mlflow.log_metric("dev_em", metric, step=epoch)
                LOGGER.info("Epoch %d - Dev EM: %.4f", epoch + 1, metric)
        torch.save(model.state_dict(), output_dir / "qa_last.pt")


def main(args: TrainArgs) -> None:
    if args.config:
        run_cfg = build_run_config(args.config)
        cfg = run_cfg.training
        data_cfg = run_cfg.data
        args.model_name_or_path = cfg.model_name_or_path
        args.output_dir = Path(cfg.output_dir)
        args.epochs = cfg.epochs
        args.batch_size = cfg.batch_size
        args.grad_accum_steps = cfg.grad_accum_steps
        args.lr = cfg.optimizer.lr
        args.weight_decay = cfg.optimizer.weight_decay
        args.warmup_steps = cfg.scheduler.warmup_steps
        args.seed = cfg.seed
        args.mixed_precision = cfg.mixed_precision
        args.margin = cfg.margin
        args.max_length = cfg.max_length
        args.mlflow_run_name = cfg.mlflow_run_name
        args.train_path = Path(data_cfg.pairwise_path)
        args.dev_path = Path(data_cfg.dev_path) if data_cfg.dev_path else None
        train(args, cfg)
    else:
        train(args)


if __name__ == "__main__":
    tyro.cli(main)
