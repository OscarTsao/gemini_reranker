"""Tests for models module."""

from __future__ import annotations

import pytest
import torch

from criteriabind.models import CrossEncoderRanker, pairwise_loss


def test_cross_encoder_ranker_initialization() -> None:
    """Test CrossEncoderRanker initialization."""
    # Use a small model for testing
    model = CrossEncoderRanker("prajjwal1/bert-tiny", dropout=0.1)

    assert model is not None
    assert hasattr(model, "encoder")
    assert hasattr(model, "head")
    assert hasattr(model, "dropout")


def test_cross_encoder_ranker_forward() -> None:
    """Test CrossEncoderRanker forward pass."""
    model = CrossEncoderRanker("prajjwal1/bert-tiny")
    model.eval()

    # Create dummy inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (batch_size,)
    assert output.dtype == torch.float32


def test_cross_encoder_ranker_with_token_type_ids() -> None:
    """Test CrossEncoderRanker with token_type_ids."""
    model = CrossEncoderRanker("prajjwal1/bert-tiny")
    model.eval()

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids, attention_mask, token_type_ids)

    assert output.shape == (batch_size,)


def test_pairwise_loss_ranknet() -> None:
    """Test pairwise loss with RankNet."""
    pos_scores = torch.tensor([1.0, 2.0, 3.0])
    neg_scores = torch.tensor([0.5, 1.5, 2.5])

    loss = pairwise_loss(pos_scores, neg_scores, "ranknet", margin=0.2)

    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_pairwise_loss_hinge() -> None:
    """Test pairwise loss with Hinge."""
    pos_scores = torch.tensor([1.0, 2.0, 3.0])
    neg_scores = torch.tensor([0.5, 1.5, 2.5])

    loss = pairwise_loss(pos_scores, neg_scores, "hinge", margin=0.2)

    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_pairwise_loss_ranknet_negative_diff() -> None:
    """Test RankNet loss when negative scores are higher."""
    pos_scores = torch.tensor([0.5, 1.5, 2.5])
    neg_scores = torch.tensor([1.0, 2.0, 3.0])  # Higher than pos

    loss = pairwise_loss(pos_scores, neg_scores, "ranknet", margin=0.2)

    # Loss should be higher when negative scores are higher
    assert loss.item() > 0.5


def test_pairwise_loss_hinge_no_violation() -> None:
    """Test Hinge loss when margin is satisfied."""
    pos_scores = torch.tensor([2.0, 3.0, 4.0])
    neg_scores = torch.tensor([0.5, 1.0, 1.5])  # Margin > 0.2 satisfied

    loss = pairwise_loss(pos_scores, neg_scores, "hinge", margin=0.2)

    # Loss should be close to 0 when margin is satisfied
    assert loss.item() < 0.1


def test_pairwise_loss_invalid_type() -> None:
    """Test pairwise loss with invalid loss type."""
    pos_scores = torch.tensor([1.0, 2.0])
    neg_scores = torch.tensor([0.5, 1.5])

    with pytest.raises(ValueError, match="Unsupported loss type"):
        pairwise_loss(pos_scores, neg_scores, "invalid", margin=0.2)  # type: ignore


def test_pairwise_loss_empty_tensors() -> None:
    """Test pairwise loss with empty tensors."""
    pos_scores = torch.tensor([])
    neg_scores = torch.tensor([])

    # Should handle empty tensors gracefully
    loss = pairwise_loss(pos_scores, neg_scores, "ranknet", margin=0.2)
    assert torch.isnan(loss)  # Mean of empty tensor is NaN


def test_cross_encoder_ranker_gradient_flow() -> None:
    """Test that gradients flow through the model."""
    model = CrossEncoderRanker("prajjwal1/bert-tiny")
    model.train()

    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones((2, 10))

    output = model(input_ids, attention_mask)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for head parameters
    for param in model.head.parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0)
