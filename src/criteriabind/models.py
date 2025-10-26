"""Backward-compatible exports for model classes."""

from .models.ranker import CrossEncoderRanker
from .models.span_extractor import SpanExtractor


__all__ = ["CrossEncoderRanker", "SpanExtractor"]
