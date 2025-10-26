"""Hydra entrypoint for Track B evidence span extractor training."""

from __future__ import annotations

import itertools
import logging
import math
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..config_schemas import AppConfig, parse_config
from ..data.collate import make_span_collate
from ..data.datasets import build_span_datasets
from ..hydra_utils import (
    auto_workers,
    enable_speed_flags,
    maybe_compile,
    resolve_device,
    set_global_seed,
)
from ..mlflow_utils import (
    get_or_create_run,
    log_artifact_dir,
    log_dataset_card,
    log_metrics,
    log_model_summary,
)
from ..models import SpanExtractor
from ..train.eval import Evaluator
from ..train.optim import build_optimizer_scheduler
from ..train.speed import Speedometer, probe_best_num_workers


LOGGER = logging.getLogger(__name__)


def _move_to_device(tensors, device: str):
    return {k: v.to(device, non_blocking=True) for k, v in tensors.items()}


def _loader_kwargs(worker_kwargs):
    kwargs = dict(worker_kwargs)
    if int(kwargs.get("num_workers", 0)) <= 0:
        kwargs.pop("prefetch_factor", None)
        kwargs["persistent_workers"] = False
    return kwargs


def _save_checkpoint(
    model: SpanExtractor,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    output_dir: Path,
    tag: str,
    metadata,
) -> Path:
    ckpt_dir = output_dir / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metadata": metadata,
        },
        ckpt_dir / "training_state.pt",
    )
    tokenizer.save_pretrained((ckpt_dir / "tokenizer").as_posix())
    return ckpt_dir


@hydra.main(config_path="../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = parse_config(cfg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not app_cfg.mlflow.run_name:
        try:
            app_cfg.mlflow.run_name = HydraConfig.get().job.name
        except ValueError:
            app_cfg.mlflow.run_name = f"{app_cfg.project_name}-run"

    try:
        from hydra.utils import get_original_cwd

        base_dir = Path(get_original_cwd())
    except Exception:
        base_dir = Path.cwd()
    output_dir = (base_dir / app_cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(app_cfg.seed)
    device, amp_dtype = resolve_device(app_cfg)
    enable_speed_flags(app_cfg)

    bundle = build_span_datasets(app_cfg)

    collate_fn = make_span_collate(bundle.tokenizer)
    worker_kwargs = auto_workers(app_cfg)
    if isinstance(app_cfg.hardware.num_workers, str) and app_cfg.hardware.num_workers == "auto":
        worker_kwargs = probe_best_num_workers(
            dataset=bundle.train,
            collate_fn=collate_fn,
            base_kwargs=worker_kwargs,
            batch_size=app_cfg.train.batch_size_per_device,
        )
    loader_kwargs = _loader_kwargs(worker_kwargs)

    train_loader = DataLoader(
        bundle.train,
        batch_size=app_cfg.train.batch_size_per_device,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=max(1, app_cfg.train.batch_size_per_device * 2),
        shuffle=False,
        collate_fn=collate_fn,
        **_loader_kwargs(worker_kwargs),
    )

    model = SpanExtractor(app_cfg.model)
    if app_cfg.hardware.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(device)
    model = maybe_compile(model, app_cfg)

    total_updates = app_cfg.train.max_steps
    optimizer, scheduler = build_optimizer_scheduler(model, app_cfg, total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda" and amp_dtype == torch.float16)

    evaluator = Evaluator(task="evidence_span", metric=app_cfg.train.metric)
    speedometer = Speedometer()

    best_metric = -math.inf
    patience_steps = 0
    best_checkpoint: Path | None = None

    if app_cfg.mlflow.autolog:
        mlflow.pytorch.autolog(log_every_n_epoch=1, disable=False)

    try:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=False)
    with get_or_create_run(app_cfg, resolved_cfg):
        log_dataset_card(bundle.card, f"{app_cfg.data.name}_span")
        mlflow.log_params(
            {
                "device": device,
                "amp_dtype": str(amp_dtype),
                "batch_size": app_cfg.train.batch_size_per_device,
                "grad_accum_steps": app_cfg.hardware.grad_accum_steps,
                "max_steps": app_cfg.train.max_steps,
                "learning_rate": app_cfg.train.lr,
            }
        )
        log_model_summary(
            f"Span extractor parameters (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        grad_accum = max(1, app_cfg.hardware.grad_accum_steps)
        global_step = 0
        micro_step = 0
        pending_loss = 0.0
        train_iter = itertools.cycle(train_loader)

        while global_step < app_cfg.train.max_steps:
            step_start = time.perf_counter()
            batch = next(train_iter)
            inputs = _move_to_device(batch["inputs"], device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            autocast_ctx = torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu",
                dtype=amp_dtype,
                enabled=device == "cuda" and amp_dtype != torch.float32,
            )
            with autocast_ctx:
                loss = model.compute_loss(inputs, start_positions, end_positions)
                loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            pending_loss += loss.item()

            micro_step += 1
            should_step = micro_step % grad_accum == 0
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                tokens = inputs.get("attention_mask")
                token_count = int(tokens.sum().item()) if tokens is not None else inputs["input_ids"].numel()
                speedometer.update(
                    batch_size=start_positions.size(0),
                    token_count=token_count,
                    step_time=time.perf_counter() - step_start,
                )

                log_metrics(step=global_step, train_loss=pending_loss * grad_accum)
                pending_loss = 0.0

                if app_cfg.train.eval_every_steps > 0 and global_step % app_cfg.train.eval_every_steps == 0:
                    metrics = evaluator.evaluate(model, val_loader, device, amp_dtype)
                    log_metrics(step=global_step, **{f"val/{k}": v for k, v in metrics.items()})
                    current_metric = metrics.get(app_cfg.train.metric, 0.0)
                    if app_cfg.train.save_top_k > 0 and evaluator.better(current_metric, best_metric):
                        best_metric = current_metric
                        patience_steps = 0
                        best_checkpoint = _save_checkpoint(
                            model,
                            bundle.tokenizer,
                            optimizer,
                            scheduler,
                            output_dir,
                            tag="best",
                            metadata={"global_step": global_step, "metric": current_metric},
                        )
                        log_artifact_dir(best_checkpoint, artifact_path="checkpoints/best")
                    else:
                        patience_steps += 1
                        if patience_steps >= app_cfg.train.early_stop_patience:
                            LOGGER.info("Early stopping triggered at step %s", global_step)
                            break

                if app_cfg.train.save_every_steps > 0 and global_step % app_cfg.train.save_every_steps == 0:
                    ckpt_path = _save_checkpoint(
                        model,
                        bundle.tokenizer,
                        optimizer,
                        scheduler,
                        output_dir,
                        tag=f"step_{global_step}",
                        metadata={"global_step": global_step},
                    )
                    log_artifact_dir(ckpt_path, artifact_path="checkpoints/periodic")

        speed_metrics = speedometer.summary()
        log_metrics(**{f"speed/{k}": v for k, v in speed_metrics.items()})
        if best_checkpoint and app_cfg.train.save_top_k > 0:
            log_artifact_dir(best_checkpoint, artifact_path="checkpoints_best")
        hydra_dir = output_dir / ".hydra"
        if hydra_dir.exists():
            log_artifact_dir(hydra_dir, artifact_path="hydra")
        LOGGER.info("Training completed. Best %s=%.4f", app_cfg.train.metric, best_metric)


if __name__ == "__main__":
    main()
