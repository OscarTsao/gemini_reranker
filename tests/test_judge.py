from __future__ import annotations

import json
import json
from pathlib import Path

import pytest
from hydra import compose, initialize

from criteriabind.cli.candidate_gen import main as candidate_main
from criteriabind.cli.judge import main as judge_main
from criteriabind.config_schemas import parse_config
from criteriabind.data.prepare_redsm5_data import main as prepare_data
from criteriabind.judge.mock_judge import MockJudge


def _run_candidate_gen(overrides: list[str]) -> None:
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="config", overrides=overrides)
    candidate_main.__wrapped__(cfg)


def test_mock_judge_outputs(tmp_path: Path) -> None:
    prepare_data()
    output_rel = f".pytest_cache/outputs/{tmp_path.name}/judge"
    overrides_common = [
        "hardware.device=cpu",
        "hardware.compile=false",
        "hardware.gradient_checkpointing=false",
        f"output_dir={output_rel}",
        "candidate_gen.k=4",
        "candidate_gen.min_char=12",
        "candidate_gen.max_char=400",
    ]
    _run_candidate_gen(overrides_common + ["split=train"])

    output_dir = Path(output_rel)

    with initialize(version_base="1.3", config_path="../conf"):
        judge_cfg = compose(
            config_name="config",
            overrides=overrides_common + ["split=train", "judge.provider=mock"],
        )
    judge_main.__wrapped__(judge_cfg)

    app_cfg = parse_config(judge_cfg)
    out_path = Path(app_cfg.judge.out_path)
    assert out_path.exists(), "expected judgments jsonl"
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "expected at least one judgment"
    first = rows[0]
    assert "best_idx" in first and "preferences" in first
    assert isinstance(first["preferences"], list)


def test_gemini_without_key_errors(tmp_path: Path) -> None:
    prepare_data()
    overrides_common = [
        "hardware.device=cpu",
        "hardware.compile=false",
        "hardware.gradient_checkpointing=false",
        f"output_dir=.pytest_cache/outputs/{tmp_path.name}/judge_gemini",
    ]
    _run_candidate_gen(overrides_common + ["split=train"])

    output_dir = Path(f".pytest_cache/outputs/{tmp_path.name}/judge_gemini")

    with initialize(version_base="1.3", config_path="../conf"):
        gemini_cfg = compose(
            config_name="config",
            overrides=overrides_common + ["split=train", "judge.provider=gemini", "judge.api_key=null"],
        )

    with pytest.raises(RuntimeError):
        judge_main.__wrapped__(gemini_cfg)


def test_gemini_with_mock(monkeypatch, tmp_path: Path) -> None:
    prepare_data()
    overrides_common = [
        "hardware.device=cpu",
        "hardware.compile=false",
        "hardware.gradient_checkpointing=false",
        f"output_dir=.pytest_cache/outputs/{tmp_path.name}/judge_gemini_ok",
    ]
    _run_candidate_gen(overrides_common + ["split=train"])

    # Patch GeminiJudge to reuse the offline MockJudge implementation.
    from criteriabind.cli import judge as judge_cli  # noqa: WPS433

    def _mock_gemini(**kwargs):  # type: ignore[no-untyped-def]
        return MockJudge()

    monkeypatch.setattr(judge_cli, "GeminiJudge", _mock_gemini)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

    output_dir = Path(f".pytest_cache/outputs/{tmp_path.name}/judge_gemini_ok")

    with initialize(version_base="1.3", config_path="../conf"):
        gemini_cfg = compose(
            config_name="config",
            overrides=overrides_common + ["split=train", "judge.provider=gemini", "judge.api_key=dummy"],
        )

    judge_main.__wrapped__(gemini_cfg)
    app_cfg = parse_config(gemini_cfg)
    out_path = Path(app_cfg.judge.out_path)
    assert out_path.exists()
