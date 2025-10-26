from __future__ import annotations

from hydra import compose, initialize

from criteriabind.config_schemas import AppConfig, parse_config


def test_config_composes() -> None:
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="config")
    app_cfg: AppConfig = parse_config(cfg)
    assert app_cfg.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert app_cfg.project_name == "gemini_reranker"
    assert app_cfg.model.from_pretrained_path.as_posix().endswith("baselines/dataaug_trial_0043/model/best")
