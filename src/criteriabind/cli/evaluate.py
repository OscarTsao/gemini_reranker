"""Evaluation CLI for ranking models."""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..config_schemas import AppConfig, parse_config
from ..data.collate import make_ranker_collate
from ..data.datasets import build_ranker_datasets
from ..hydra_utils import enable_speed_flags, resolve_device, set_global_seed
from ..mlflow_utils import get_or_create_run, log_dataset_card, log_metrics
from ..models.ranker import CrossEncoderRanker
from ..train.eval import Evaluator


LOGGER = logging.getLogger(__name__)


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

    bundle = build_ranker_datasets(app_cfg)
    collate_fn = make_ranker_collate(bundle.tokenizer)
    val_loader = torch.utils.data.DataLoader(
        bundle.val,
        batch_size=max(1, app_cfg.train.batch_size_per_device * 2),
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CrossEncoderRanker(app_cfg.model)
    model.to(device)
    model.eval()

    evaluator = Evaluator(task="criteria_ranker", metric=app_cfg.train.metric)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    with get_or_create_run(app_cfg, resolved_cfg):
        log_dataset_card(bundle.card, app_cfg.data.name)
        metrics = evaluator.evaluate(model, val_loader, device, amp_dtype)
        log_metrics(**{f"eval/{k}": v for k, v in metrics.items()})
        LOGGER.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
