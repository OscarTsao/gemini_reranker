"""Training loop for criteria matching cross-encoder ranker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import tyro
import mlflow

from .config import build_run_config, TrainingConfig
from .io_utils import read_jsonl
from .logging_utils import get_logger
from .metrics import pairwise_accuracy
from .mlflow_utils import mlflow_run
from .models import CrossEncoderRanker, load_cross_encoder, pairwise_loss
from .schemas import PairwiseRow
from .seed import seed_everything

LOGGER = get_logger(__name__)


class PairwiseDataset(Dataset):
    """Simple dataset for pairwise ranking."""

    def __init__(self, rows: Sequence[PairwiseRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PairwiseRow:
        return self.rows[index]


def load_dataset(path: Optional[Path]) -> List[PairwiseRow]:
    if not path or not path.exists():
        return []
    return [PairwiseRow(**row) for row in read_jsonl(path)]


def collate_fn(batch: Sequence[PairwiseRow], tokenizer, max_length: int):
    criteria = [row.criterion for row in batch]
    pos_texts = [row.pos for row in batch]
    neg_texts = [row.neg for row in batch]
    pos_inputs = tokenizer(
        criteria,
        pos_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    neg_inputs = tokenizer(
        criteria,
        neg_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {"pos": pos_inputs, "neg": neg_inputs}


@dataclass
class TrainArgs:
    """CLI arguments."""

    config: Optional[Path] = None
    pairs_path: Path = Path("data/pairs/redsm5_criteria_train.jsonl")
    dev_path: Optional[Path] = Path("data/pairs/redsm5_criteria_dev.jsonl")
    output_dir: Path = Path("data/models/redsm5_criteria")
    model_name_or_path: str = "baselines/dataaug_trial_0043/model/best"
    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_length: int = 512
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 42
    mixed_precision: str = "bf16"
    loss_type: str = "ranknet"
    margin: float = 0.2
    mlflow_run_name: Optional[str] = None
    resume_from: Optional[Path] = None


def evaluate(model: CrossEncoderRanker, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores_pos: List[float] = []
    scores_neg: List[float] = []
    with torch.no_grad():
        for batch in data_loader:
            pos_inputs = {k: v.to(device) for k, v in batch["pos"].items()}
            neg_inputs = {k: v.to(device) for k, v in batch["neg"].items()}
            pos_scores = model(**pos_inputs)
            neg_scores = model(**neg_inputs)
            scores_pos.extend(pos_scores.cpu().tolist())
            scores_neg.extend(neg_scores.cpu().tolist())
    model.train()
    if not scores_pos:
        return 0.0
    return pairwise_accuracy(scores_pos, scores_neg)


def save_checkpoint(
    model: CrossEncoderRanker,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(
    model: CrossEncoderRanker,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: Path,
) -> int:
    LOGGER.info("Resuming from checkpoint %s", path)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint.get("step", 0)


def train_loop(args: TrainArgs, cfg: Optional[TrainingConfig] = None) -> None:
    seed_everything(args.seed if cfg is None else cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_dataset(args.pairs_path)
    dev_rows = load_dataset(args.dev_path)
    LOGGER.info("Training pairs: %d, Dev pairs: %d", len(rows), len(dev_rows))
    dataset = PairwiseDataset(rows)
    bundle = load_cross_encoder(args.model_name_or_path if cfg is None else cfg.model_name_or_path)
    model = bundle.model.to(device)
    tokenizer = bundle.tokenizer
    max_length = args.max_length if cfg is None else cfg.max_length
    if bundle.max_length is not None:
        max_length = min(max_length, bundle.max_length)
        LOGGER.info("Capping sequence length to %d based on checkpoint", max_length)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size if cfg is None else cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
    )
    dev_loader = None
    if dev_rows:
        dev_loader = DataLoader(
            PairwiseDataset(dev_rows),
            batch_size=args.batch_size if cfg is None else cfg.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr if cfg is None else cfg.optimizer.lr,
        weight_decay=args.weight_decay if cfg is None else cfg.optimizer.weight_decay,
    )
    optimizer.zero_grad()
    total_steps = max(
        1,
        len(data_loader)
        * (args.epochs if cfg is None else cfg.epochs)
        // (args.grad_accum_steps if cfg is None else cfg.grad_accum_steps),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps if cfg is None else cfg.scheduler.warmup_steps,
        num_training_steps=total_steps,
    )

    precision = (args.mixed_precision if cfg is None else cfg.mixed_precision).lower()
    use_autocast = precision in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_from)

    best_metric = 0.0
    output_dir = args.output_dir if cfg is None else Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accumulation_steps = args.grad_accum_steps if cfg is None else cfg.grad_accum_steps
    total_epochs = args.epochs if cfg is None else cfg.epochs
    global_step = start_step

    with mlflow_run(
        args.mlflow_run_name if cfg is None else cfg.mlflow_run_name,
        params={"epochs": total_epochs, "lr": optimizer.param_groups[0]["lr"]},
    ):
        for epoch in range(total_epochs):
            for batch_idx, batch in enumerate(data_loader):
                pos_inputs = {k: v.to(device) for k, v in batch["pos"].items()}
                neg_inputs = {k: v.to(device) for k, v in batch["neg"].items()}
                autocast_dtype = torch.bfloat16 if precision == "bf16" else None
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    pos_scores = model(**pos_inputs)
                    neg_scores = model(**neg_inputs)
                    loss = pairwise_loss(
                        pos_scores,
                        neg_scores,
                        loss_type=args.loss_type if cfg is None else cfg.loss_type,
                        margin=args.margin if cfg is None else cfg.margin,
                    )
                    loss = loss / accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
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
                    global_step += 1
            if dev_loader:
                metric = evaluate(model, dev_loader, device)
                LOGGER.info("Epoch %d - Dev pairwise accuracy: %.4f", epoch + 1, metric)
                mlflow.log_metric("dev_pairwise_accuracy", metric, step=epoch)
                if metric > best_metric:
                    best_metric = metric
                    save_checkpoint(model, optimizer, scheduler, global_step, output_dir / "best.ckpt")
        save_checkpoint(model, optimizer, scheduler, global_step, output_dir / "last.ckpt")


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
        args.max_length = cfg.max_length
        args.lr = cfg.optimizer.lr
        args.weight_decay = cfg.optimizer.weight_decay
        args.warmup_steps = cfg.scheduler.warmup_steps
        args.seed = cfg.seed
        args.mixed_precision = cfg.mixed_precision
        args.loss_type = cfg.loss_type
        args.margin = cfg.margin
        args.mlflow_run_name = cfg.mlflow_run_name
        args.resume_from = Path(cfg.resume_from) if cfg.resume_from else None
        args.pairs_path = Path(data_cfg.pairwise_path)
        args.dev_path = Path(data_cfg.dev_path) if data_cfg.dev_path else None
        train_loop(args, cfg)
    else:
        train_loop(args)


if __name__ == "__main__":
    tyro.cli(main)
