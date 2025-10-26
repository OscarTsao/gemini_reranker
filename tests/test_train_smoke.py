from __future__ import annotations

import shutil
from pathlib import Path

from hydra import compose, initialize
from mlflow.tracking import MlflowClient

from criteriabind.data.prepare_redsm5_data import main as prepare_data
from criteriabind.train.train_criteria_ranker import main as train_main


def test_training_smoke(tmp_path: Path) -> None:
    prepare_data()
    tracking_rel = f".pytest_cache/mlruns/{tmp_path.name}/mlflow.db"
    artifacts_rel = f".pytest_cache/mlruns/{tmp_path.name}"

    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "hardware.device=cpu",
                "hardware.compile=false",
                "mlflow.autolog=false",
                "train.max_steps=1",
                "train.eval_every_steps=1",
                "train.save_every_steps=0",
                "train.save_top_k=0",
                "data.max_samples=4",
                "output_dir=test_outputs",
                f"mlflow.tracking_uri=sqlite:///{tracking_rel}",
                f"mlflow.artifact_root={artifacts_rel}",
            ],
        )

    Path(tracking_rel).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts_rel).mkdir(parents=True, exist_ok=True)

    train_main.__wrapped__(cfg)

    tracking_path = Path(tracking_rel)
    assert tracking_path.exists()
    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_rel}")
    experiment = client.get_experiment_by_name("gemini_reranker")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs, "Expected at least one MLflow run"

    outputs_root = Path("outputs")
    if outputs_root.exists():
        shutil.rmtree(outputs_root)
    shutil.rmtree(Path(artifacts_rel).parent, ignore_errors=True)
