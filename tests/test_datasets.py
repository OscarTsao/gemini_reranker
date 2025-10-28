from __future__ import annotations

import shutil
from pathlib import Path

from hydra import compose, initialize

from criteriabind.config_schemas import parse_config
from criteriabind.data.collate import BucketBatchSampler, make_ranker_collate
from criteriabind.data.datasets import build_ranker_datasets
from criteriabind.data.prepare_redsm5_data import main as prepare_data


def test_ranker_dataset_caching(tmp_path: Path, monkeypatch) -> None:
    prepare_data()
    cache_rel = f".pytest_cache/datasets/{tmp_path.name}"
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=["data.max_samples=10", f"data.cache_dir={cache_rel}"],
        )
    app_cfg = parse_config(cfg)
    bundle = build_ranker_datasets(app_cfg)
    assert len(bundle.train) > 0
    cache_dir = Path(cache_rel)
    cache_files = list(cache_dir.glob("ranker_train_*.pt"))
    assert cache_files, "expected cache artifact"

    collate = make_ranker_collate(bundle.tokenizer)
    sample_indices = list(range(min(8, len(bundle.train))))
    sampler = BucketBatchSampler(bundle.train.lengths, batch_size=4, shuffle=False, drop_last=False)
    batches = list(iter(sampler))
    assert batches, "sampler should yield batches"
    batch = [bundle.train[idx] for idx in sample_indices]
    collated = collate(batch)
    assert "pos_inputs" in collated and "neg_inputs" in collated
    assert "weights" in collated
    assert collated["weights"].shape[0] == len(batch)
    shutil.rmtree(cache_dir, ignore_errors=True)
