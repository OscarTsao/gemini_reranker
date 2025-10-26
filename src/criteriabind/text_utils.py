"""Text processing utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence


try:
    from nltk import sent_tokenize
except ImportError:  # pragma: no cover - fallback used rarely
    sent_tokenize = None

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and normalise quotes."""
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def sentence_tokenize(text: str) -> list[str]:
    """Tokenise text into sentences with NLTK fallback."""
    text = normalize_whitespace(text)
    if sent_tokenize:
        try:
            sentences = sent_tokenize(text)
            if sentences:
                return [normalize_whitespace(sentence) for sentence in sentences]
        except LookupError:
            # Punkt not downloaded; fallback to regex split.
            pass
    sentences = SENTENCE_SPLIT_REGEX.split(text)
    return [normalize_whitespace(sentence) for sentence in sentences if sentence]


def chunk_text(
    text: str,
    max_tokens: int,
    stride: int | None = None,
    token_pattern: str = r"\w+|\S",
) -> list[str]:
    """Chunk text into overlapping windows based on regex token counts."""
    tokens = re.findall(token_pattern, text)
    if not tokens:
        return [text] if text else []
    stride = stride or max(max_tokens // 2, 1)
    windows: list[str] = []
    for start in range(0, len(tokens), stride):
        end = min(start + max_tokens, len(tokens))
        if start >= end:
            break
        window = " ".join(tokens[start:end])
        windows.append(window)
        if end == len(tokens):
            break
    return windows


def extract_spans(text: str, sentence: str, span_lengths: Sequence[int]) -> list[tuple[int, int, str]]:
    """Generate candidate spans within a sentence, returning (start, end, span_text)."""
    spans: list[tuple[int, int, str]] = []
    sentence = normalize_whitespace(sentence)
    base_index = text.find(sentence)
    if base_index == -1:
        return spans
    tokens = sentence.split()
    if not tokens:
        return spans
    for length in span_lengths:
        if length <= 0:
            continue
        for start in range(len(tokens)):
            end = start + length
            if end > len(tokens):
                break
            span_tokens = tokens[start:end]
            span_text = " ".join(span_tokens)
            span_start = _find_substring(text, span_text, base_index)
            if span_start is None:
                continue
            spans.append((span_start, span_start + len(span_text), span_text))
    return spans


def _find_substring(text: str, substring: str, offset: int) -> int | None:
    """Locate substring within text starting from offset."""
    index = text.find(substring, offset)
    return index if index != -1 else None
