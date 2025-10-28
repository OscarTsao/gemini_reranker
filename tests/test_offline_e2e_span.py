from __future__ import annotations

import json
import shutil
from pathlib import Path

from hydra import compose, initialize

from criteriabind.config_schemas import parse_config
from criteriabind.data.prepare_redsm5_data import main as prepare_data
from criteriabind.train.train_evidence_span import main as span_train


def test_offline_span_pipeline(tmp_path: Path) -> None:
    prepare_data()
    tracking_rel = f".pytest_cache/mlruns/{tmp_path.name}/span.db"
    artifacts_rel = f".pytest_cache/mlruns/{tmp_path.name}/span_artifacts"
    output_rel = f".pytest_cache/outputs/{tmp_path.name}/span"

    Path(tracking_rel).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts_rel).mkdir(parents=True, exist_ok=True)

    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "hardware.device=cpu",
                "hardware.compile=false",
                "hardware.gradient_checkpointing=false",
                "mlflow.autolog=false",
                "data.max_samples=6",
                f"mlflow.tracking_uri=sqlite:///{tracking_rel}",
                f"mlflow.artifact_root={artifacts_rel}",
                f"output_dir={output_rel}",
                "train=span_fast",
                "train.max_steps=4",
                "train.eval_every_steps=4",
                "train.save_top_k=0",
                "train.save_every_steps=0",
                "train.benchmark_steps=4",
            ],
        )

    span_train.__wrapped__(cfg)

    app_cfg = parse_config(cfg)
    output_dir = Path(app_cfg.output_dir)
    speed_path = output_dir / "speed.json"
    assert speed_path.exists()
    payload = json.loads(speed_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["samples_per_sec"] > 0

    shutil.rmtree(Path(artifacts_rel).parent, ignore_errors=True)
    shutil.rmtree(output_dir.parent, ignore_errors=True)
