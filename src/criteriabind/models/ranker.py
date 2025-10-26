"""Cross-encoder ranker built on top of Hugging Face transformer encoders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from transformers import AutoConfig, AutoModel


LOGGER = logging.getLogger(__name__)


class CrossEncoderRanker(nn.Module):
    """Pairwise cross-encoder with optional baseline checkpoint support."""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        pretrained_path = model_cfg.from_pretrained_path or model_cfg.model_name
        baseline_dir = Path(pretrained_path)
        if baseline_dir.is_dir() and (baseline_dir / "model.pt").exists():
            self._init_from_baseline(baseline_dir, model_cfg)
        else:
            config = AutoConfig.from_pretrained(pretrained_path)
            self.encoder = AutoModel.from_pretrained(pretrained_path, config=config)
            hidden_size = config.hidden_size
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.dropout_layer: Optional[nn.Dropout] = None
            self.use_baseline_head = False
        self.gradient_checkpointing = False

    def _init_from_baseline(self, baseline_dir: Path, model_cfg) -> None:
        config_path = baseline_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Baseline checkpoint at {baseline_dir} missing config.yaml"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        model_section = cfg.get("model", {})
        pretrained_name = model_section.get("pretrained_model_name", model_cfg.model_name)
        dropout = float(model_section.get("classifier_dropout", 0.0))
        hidden_sizes = [int(size) for size in model_section.get("classifier_hidden_sizes", [])]
        num_labels = int(model_section.get("num_labels", 2))

        config = AutoConfig.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name, config=config)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = size
        layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)
        self.use_baseline_head = True
        state = torch.load(baseline_dir / "model.pt", map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            LOGGER.warning("Missing parameters when loading baseline checkpoint: %s", missing)
        if unexpected:
            LOGGER.warning("Unexpected parameters when loading baseline checkpoint: %s", unexpected)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        encoder = getattr(self, "encoder", getattr(self, "bert", None))
        if encoder is None:
            raise RuntimeError("Encoder not initialised")
        outputs = encoder(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        if self.dropout_layer is not None:
            pooled = self.dropout_layer(pooled)
        if self.use_baseline_head:
            logits = self.classifier(pooled)
            if logits.dim() == 2 and logits.size(-1) >= 1:
                logits = logits[..., -1]
            return logits.squeeze(-1)
        logits = self.scorer(pooled)
        return logits.squeeze(-1)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        margin: float = 0.0,
    ) -> torch.Tensor:
        pos_scores = self.forward(batch["pos_inputs"])
        neg_scores = self.forward(batch["neg_inputs"])
        if margin > 0:
            losses = torch.relu(margin - (pos_scores - neg_scores))
        else:
            losses = F.softplus(-(pos_scores - neg_scores))
        return losses.mean()

    def enable_gradient_checkpointing(self) -> None:
        encoder = getattr(self, "encoder", getattr(self, "bert", None))
        if encoder is not None and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
            self.gradient_checkpointing = True
            LOGGER.info("Enabled gradient checkpointing for ranker.")

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else self.named_parameters()
        if trainable_only:
            return sum(p.numel() for _, p in params if p.requires_grad)
        return sum(p.numel() for p in params)
