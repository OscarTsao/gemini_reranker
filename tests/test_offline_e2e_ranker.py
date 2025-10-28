from __future__ import annotations

import json
import shutil
from pathlib import Path

from hydra import compose, initialize

from criteriabind.cli.candidate_gen import main as candidate_main
from criteriabind.cli.judge import main as judge_main
from criteriabind.cli.pair_builder import main as pair_main
from criteriabind.config_schemas import parse_config
from criteriabind.data.prepare_redsm5_data import main as prepare_data
from criteriabind.train.train_criteria_ranker import main as train_main


def test_offline_ranker_pipeline(tmp_path: Path) -> None:
    prepare_data()
    tracking_rel = f".pytest_cache/mlruns/{tmp_path.name}/pipeline.db"
    artifacts_rel = f".pytest_cache/mlruns/{tmp_path.name}/artifacts"
    output_rel = f".pytest_cache/outputs/{tmp_path.name}/ranker"

    overrides_common = [
        "hardware.device=cpu",
        "hardware.compile=false",
        "hardware.gradient_checkpointing=false",
        "mlflow.autolog=false",
        "data.max_samples=8",
        f"mlflow.tracking_uri=sqlite:///{tracking_rel}",
        f"mlflow.artifact_root={artifacts_rel}",
        f"output_dir={output_rel}",
    ]

    Path(tracking_rel).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts_rel).mkdir(parents=True, exist_ok=True)

    with initialize(version_base="1.3", config_path="../conf"):
        candidate_cfg = compose(
            config_name="config",
            overrides=overrides_common
            + [
                "split=train",
                "candidate_gen.k=4",
                "candidate_gen.min_char=12",
                "candidate_gen.max_char=300",
            ],
        )
        candidate_main.__wrapped__(candidate_cfg)

        judge_cfg = compose(
            config_name="config",
            overrides=overrides_common + ["split=train", "judge.provider=mock"],
        )
        judge_main.__wrapped__(judge_cfg)

        pair_cfg = compose(
            config_name="config",
            overrides=overrides_common + ["split=train", "pair_builder.mode=both"],
        )
        pair_main.__wrapped__(pair_cfg)

        train_cfg = compose(
            config_name="config",
            overrides=overrides_common
            + [
                "train.max_steps=4",
                "train.eval_every_steps=4",
                "train.save_every_steps=0",
                "train.save_top_k=1",
                "train.benchmark_steps=4",
            ],
        )

    train_main.__wrapped__(train_cfg)

    app_cfg = parse_config(train_cfg)
    output_dir = Path(app_cfg.output_dir)
    speed_path = output_dir / "speed.json"
    assert speed_path.exists(), f"expected speed.json at {speed_path}"
    speed_payload = json.loads(speed_path.read_text(encoding="utf-8"))
    assert speed_payload["metrics"]["samples_per_sec"] > 0
    assert "dataloader_wait_ratio" in speed_payload["metrics"]

    shutil.rmtree(Path(artifacts_rel).parent, ignore_errors=True)
    shutil.rmtree(output_dir.parent, ignore_errors=True)
