"""Tests for config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from criteriabind.config import (
    DataConfig,
    JudgeConfig,
    RunConfig,
    TrainingConfig,
    build_run_config,
    load_yaml_config,
    merge_dicts,
    update_dataclass,
)


def test_judge_config_defaults() -> None:
    """Test JudgeConfig default values."""
    config = JudgeConfig()
    assert config.model == "gemini-2.5-flash"
    assert config.rubric_version == "clinical_rubric_v1"
    assert config.enable_two_pass is True
    assert config.max_group_size == 5


def test_judge_config_to_dict() -> None:
    """Test JudgeConfig serialization."""
    config = JudgeConfig(model="gemini-2.0-flash", retries=5)
    data = config.to_dict()
    assert isinstance(data, dict)
    assert data["model"] == "gemini-2.0-flash"
    assert data["retries"] == 5


def test_training_config_defaults() -> None:
    """Test TrainingConfig default values."""
    config = TrainingConfig(model_name_or_path="bert-base-uncased", output_dir="output")
    assert config.epochs == 3
    assert config.batch_size == 8
    assert config.seed == 42
    assert config.loss_type == "ranknet"


def test_training_config_to_dict() -> None:
    """Test TrainingConfig serialization."""
    config = TrainingConfig(
        model_name_or_path="bert-base-uncased",
        output_dir="output",
        epochs=5,
        batch_size=16,
    )
    data = config.to_dict()
    assert isinstance(data, dict)
    assert data["epochs"] == 5
    assert data["batch_size"] == 16
    assert "optimizer" in data


def test_data_config_defaults() -> None:
    """Test DataConfig default values."""
    config = DataConfig()
    assert config.raw_path == "data/raw/train.jsonl"
    assert config.dev_path is None


def test_run_config() -> None:
    """Test RunConfig composition."""
    train_cfg = TrainingConfig(model_name_or_path="bert", output_dir="out")
    data_cfg = DataConfig()
    judge_cfg = JudgeConfig()

    run_cfg = RunConfig(training=train_cfg, data=data_cfg, judge=judge_cfg)
    assert run_cfg.training == train_cfg
    assert run_cfg.data == data_cfg
    assert run_cfg.judge == judge_cfg

    data = run_cfg.to_dict()
    assert "training" in data
    assert "data" in data
    assert "judge" in data


def test_load_yaml_config(tmp_path: Path) -> None:
    """Test loading YAML config file."""
    config_file = tmp_path / "config.yaml"
    config_data = {"key1": "value1", "key2": 123, "nested": {"key3": "value3"}}

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    loaded = load_yaml_config(config_file)
    assert loaded["key1"] == "value1"
    assert loaded["key2"] == 123
    assert loaded["nested"]["key3"] == "value3"


def test_load_yaml_config_invalid(tmp_path: Path) -> None:
    """Test loading invalid YAML config."""
    config_file = tmp_path / "config.yaml"

    # Write non-dict YAML (list instead)
    with open(config_file, "w") as f:
        f.write("- item1\n- item2\n")

    with pytest.raises(ValueError, match="must be a dictionary"):
        load_yaml_config(config_file)


def test_build_run_config(tmp_path: Path) -> None:
    """Test building RunConfig from YAML."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "training": {
            "model_name_or_path": "bert-base-uncased",
            "output_dir": "output",
            "epochs": 5,
            "batch_size": 16,
        },
        "data": {"raw_path": "data/custom.jsonl"},
        "judge": {"model": "gemini-2.0-flash"},
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    run_cfg = build_run_config(config_file)
    assert run_cfg.training.epochs == 5
    assert run_cfg.training.batch_size == 16
    assert run_cfg.data.raw_path == "data/custom.jsonl"
    assert run_cfg.judge.model == "gemini-2.0-flash"


def test_update_dataclass() -> None:
    """Test updating dataclass fields."""
    config = JudgeConfig()
    assert config.model == "gemini-2.5-flash"

    update_dataclass(config, {"model": "gemini-2.0-flash", "retries": 5})
    assert config.model == "gemini-2.0-flash"
    assert config.retries == 5


def test_update_dataclass_invalid_field() -> None:
    """Test updating dataclass with invalid field."""
    config = JudgeConfig()

    # Should not raise, just ignore invalid fields
    update_dataclass(config, {"nonexistent_field": "value"})


def test_merge_dicts() -> None:
    """Test merging dictionaries."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    dict3 = {"c": 5, "d": 6}

    merged = merge_dicts(dict1, dict2, dict3)
    assert merged["a"] == 1
    assert merged["b"] == 3  # Later dict overwrites
    assert merged["c"] == 5  # Latest dict wins
    assert merged["d"] == 6


def test_merge_dicts_empty() -> None:
    """Test merging empty dictionaries."""
    merged = merge_dicts()
    assert merged == {}

    merged = merge_dicts({}, {}, {})
    assert merged == {}
