"""Tests for io_utils module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from criteriabind.io_utils import (
    batched,
    read_jsonl,
    read_jsonl_with_parser,
    write_jsonl,
    write_sharded_jsonl,
)
from criteriabind.schemas import Sample


def test_read_jsonl(tmp_path: Path) -> None:
    """Test reading JSONL file."""
    test_file = tmp_path / "test.jsonl"
    test_data = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]

    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    result = list(read_jsonl(test_file))
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["text"] == "world"


def test_read_jsonl_empty_lines(tmp_path: Path) -> None:
    """Test reading JSONL with empty lines."""
    test_file = tmp_path / "test.jsonl"

    with open(test_file, "w") as f:
        f.write('{"id": "1"}\n')
        f.write("\n")
        f.write('{"id": "2"}\n')
        f.write("   \n")
        f.write('{"id": "3"}\n')

    result = list(read_jsonl(test_file))
    assert len(result) == 3


def test_write_jsonl(tmp_path: Path) -> None:
    """Test writing JSONL file."""
    test_file = tmp_path / "output.jsonl"
    test_data = [{"id": "1", "value": 10}, {"id": "2", "value": 20}]

    write_jsonl(test_file, test_data)

    assert test_file.exists()
    result = list(read_jsonl(test_file))
    assert len(result) == 2
    assert result[0]["value"] == 10


def test_write_jsonl_with_schema(tmp_path: Path) -> None:
    """Test writing JSONL with Schema objects."""
    test_file = tmp_path / "output.jsonl"
    samples = [
        Sample(id="1", note_text="text1", criteria=["a", "b"]),
        Sample(id="2", note_text="text2", criteria=["c", "d"]),
    ]

    write_jsonl(test_file, samples)

    assert test_file.exists()
    result = list(read_jsonl(test_file))
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["criteria"] == ["c", "d"]


def test_write_jsonl_creates_parent_dir(tmp_path: Path) -> None:
    """Test that write_jsonl creates parent directories."""
    test_file = tmp_path / "nested" / "dir" / "output.jsonl"
    test_data = [{"id": "1"}]

    write_jsonl(test_file, test_data)

    assert test_file.exists()
    assert test_file.parent.exists()


def test_write_sharded_jsonl(tmp_path: Path) -> None:
    """Test writing sharded JSONL files."""
    pattern = str(tmp_path / "shard_{shard:03d}.jsonl")
    test_data = [{"id": str(i)} for i in range(10)]

    paths = write_sharded_jsonl(pattern, test_data, shard_size=3)

    assert len(paths) == 4  # 10 items / 3 per shard = 4 shards
    assert Path(paths[0]).exists()

    # Check first shard has 3 items
    shard0 = list(read_jsonl(paths[0]))
    assert len(shard0) == 3

    # Check last shard has 1 item
    shard3 = list(read_jsonl(paths[3]))
    assert len(shard3) == 1


def test_batched() -> None:
    """Test batched function."""
    data = list(range(10))
    batches = list(batched(data, 3))

    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]


def test_batched_invalid_size() -> None:
    """Test batched with invalid batch size."""
    data = [1, 2, 3]

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(batched(data, 0))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(batched(data, -1))


def test_read_jsonl_with_parser(tmp_path: Path) -> None:
    """Test reading JSONL with custom parser."""
    test_file = tmp_path / "test.jsonl"
    test_data = [
        {"id": "1", "note_text": "text1", "criteria": ["a"]},
        {"id": "2", "note_text": "text2", "criteria": ["b"]},
    ]

    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    def parser(row: dict) -> Sample:
        return Sample(**row)

    samples = list(read_jsonl_with_parser(test_file, parser))
    assert len(samples) == 2
    assert isinstance(samples[0], Sample)
    assert samples[0].id == "1"
    assert samples[1].criteria == ["b"]


def test_read_jsonl_malformed(tmp_path: Path) -> None:
    """Test reading JSONL with malformed JSON."""
    test_file = tmp_path / "bad.jsonl"

    with open(test_file, "w") as f:
        f.write('{"id": "1"}\n')
        f.write('{"id": "2", invalid json}\n')
        f.write('{"id": "3"}\n')

    with pytest.raises(ValueError):
        list(read_jsonl(test_file))
