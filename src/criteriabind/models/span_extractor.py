"""Span extraction head for evidence localization."""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


LOGGER = logging.getLogger(__name__)


class SpanExtractor(nn.Module):
    """Question-answering style span predictor."""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        pretrained_path = model_cfg.from_pretrained_path or model_cfg.model_name
        config = AutoConfig.from_pretrained(pretrained_path)
        self.encoder = AutoModel.from_pretrained(pretrained_path, config=config)
        hidden_size = config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(**inputs)
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

    def compute_loss(
        self,
        inputs: dict[str, torch.Tensor],
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> torch.Tensor:
        start_logits, end_logits = self.forward(inputs)
        loss_fn = nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        return (start_loss + end_loss) / 2

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            LOGGER.info("Enabled gradient checkpointing for span extractor.")
