"""I/O helpers for JSONL datasets."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TypeVar, Callable

import ujson

from .schemas import SchemaEncoder

T = TypeVar("T")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield dictionaries from a JSONL file.

    Args:
        path: Path to JSONL file.

    Yields:
        Dictionary for each line.

    Raises:
        ValueError: If JSON parsing fails.
        FileNotFoundError: If file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path_obj.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ujson.loads(line)
            except ValueError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[dict | SchemaEncoder]) -> None:
    """Write dictionaries or schema objects to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, SchemaEncoder):
                handle.write(f"{row.to_json()}\n")
            else:
                handle.write(f"{ujson.dumps(row)}\n")


def write_sharded_jsonl(
    path_pattern: str,
    rows: Sequence[dict | SchemaEncoder],
    shard_size: int,
) -> List[str]:
    """Write rows into sharded JSONL files following the provided pattern.

    Args:
        path_pattern: Pattern with ``{shard}`` placeholder, e.g.
            ``data/pairs/train_{shard:03d}.jsonl``.
        rows: Sequence of rows to write.
        shard_size: Maximum number of rows per shard.

    Returns:
        List of file paths created.
    """
    total = len(rows)
    num_shards = math.ceil(total / shard_size) if shard_size > 0 else 1
    paths: List[str] = []
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(total, (shard_idx + 1) * shard_size)
        shard_rows = rows[start:end]
        shard_path = path_pattern.format(shard=shard_idx)
        write_jsonl(shard_path, shard_rows)
        paths.append(shard_path)
    return paths


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Chunk an iterable into lists of ``batch_size``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def read_jsonl_with_parser(path: str | Path, parser: Callable[[dict], T]) -> Iterator[T]:
    """Read a JSONL file and parse each row with the provided parser callable."""
    for row in read_jsonl(path):
        yield parser(row)
