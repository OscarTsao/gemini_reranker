"""Model architectures for ranking and QA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class CrossEncoderRanker(nn.Module):
    """Cross-encoder ranker with a scalar head."""

    def __init__(self, model_name_or_path: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.head(self.dropout(pooled))
        return logits.squeeze(-1)


def pairwise_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    loss_type: Literal["ranknet", "hinge"],
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute pairwise loss between positive and negative scores."""
    if loss_type == "ranknet":
        diff = pos_scores - neg_scores
        return torch.nn.functional.softplus(-diff).mean()
    if loss_type == "hinge":
        diff = pos_scores - neg_scores
        return torch.nn.functional.relu(margin - diff).mean()
    raise ValueError(f"Unsupported loss type {loss_type}")


@dataclass
class CrossEncoderBundle:
    model: CrossEncoderRanker
    tokenizer: PreTrainedTokenizerBase


def load_cross_encoder(model_name_or_path: str) -> CrossEncoderBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = CrossEncoderRanker(model_name_or_path)
    return CrossEncoderBundle(model=model, tokenizer=tokenizer)


class QASpanModel(nn.Module):
    """Wrapper around AutoModelForQuestionAnswering with pairwise margin loss."""

    def __init__(self, model_name_or_path: str, margin: float = 0.3) -> None:
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.margin = margin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        neg_start_positions: Optional[torch.Tensor] = None,
        neg_end_positions: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss
        if (
            neg_start_positions is not None
            and neg_end_positions is not None
            and neg_mask is not None
            and neg_mask.any()
        ):
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            pos_score = start_logits.gather(1, start_positions.unsqueeze(1)) + end_logits.gather(
                1, end_positions.unsqueeze(1)
            )
            neg_score = start_logits.gather(1, neg_start_positions.unsqueeze(1)) + end_logits.gather(
                1, neg_end_positions.unsqueeze(1)
            )
            pos_score = pos_score.squeeze(-1)
            neg_score = neg_score.squeeze(-1)
            masked_diff = pos_score[neg_mask] - neg_score[neg_mask]
            margin_loss = torch.nn.functional.relu(self.margin - masked_diff).mean()
            loss = loss + margin_loss
        return {"loss": loss, "start_logits": outputs.start_logits, "end_logits": outputs.end_logits}
