# Gemini Reranker

Hydra-driven pipelines for preference-based reranking and span extraction using Gemini-provided labels and MLflow tracking. The repository ships with an offline baseline checkpoint, deterministic demo data, and tooling to run the full candidate generation ➜ judging ➜ dataset ➜ training ➜ inference workflow on CPU in minutes.

## Quickstart

```bash
make setup                   # pip install -e ".[dev]" and NLTK punkt
docker pull nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04  # optional CUDA base
make demo-data               # populate demo_data/redsm5/
make train-criteria          # 1 GPU/CPU-friendly RankNet loop
```

The command above logs into `mlflow.db` (SQLite) and emits artefacts under `outputs/`. To bring up a local MLflow UI:

```bash
make mlflow-up    # serves http://127.0.0.1:5000 with ./mlruns/ artefacts
```

## Hydra CLI Examples

All entrypoints accept Hydra overrides. Common patterns:

```bash
python -m criteriabind.train.train_criteria_ranker \
  train.max_steps=300 \
  hardware.device=cpu \
  train.save_top_k=0 \
  mlflow.tracking_uri=sqlite:///tmp/mlflow.db

python -m criteriabind.train.train_evidence_span \
  train=span_fast hydra.job.name=span_demo \
  hardware.device=auto \
  data.max_samples=200

python -m criteriabind.cli.judge split=train judge.provider=mock
python -m criteriabind.cli.pair_builder split=train
python -m criteriabind.cli.infer split=test
python -m criteriabind.cli.evaluate
```

Hydra outputs live in `outputs/%Y-%m-%d/%H-%M-%S-job`. Override `output_dir` if you need deterministic paths.

## Offline Baseline

`baselines/dataaug_trial_0043/model/best/` contains the Optuna trial-0043 DeBERTa-base classifier (`model.pt`) and `config.yaml` describing its MLP head. The default model config loads these weights via `from_pretrained_path`, ensuring training, inference, and evaluation start from the established baseline without external downloads. Tokenisation still relies on `microsoft/deberta-base` and is cached under `~/.cache/huggingface`.

## Tooling & Layout

- `conf/` – Hydra configuration tree (defaults, hardware, MLflow, data, models, judges).
- `src/criteriabind/` – package code (data loaders, models, training, CLI, utilities).
- `demo_data/` – synthetic ReDSM5-style JSONL splits created by `make demo-data`.
- `baselines/` – frozen baseline checkpoint and augmentation artefacts.
- `tests/` – pytest suite covering configs, hardware flags, dataset caching, and training smoke runs.
- `requirements.txt` / `environment.yml` / `Dockerfile` – reproducible environment manifests.

### Make Targets

| Target            | Description |
|-------------------|-------------|
| `make setup`      | Install editable package with dev extras and download NLTK punkt |
| `make demo-data`  | Generate the tiny ReDSM5 demo splits |
| `make judge`      | Run the mock Gemini judge (defaults to offline provider) |
| `make pairs`      | Build pairwise datasets from judged outputs |
| `make train-criteria` | Launch the ranker training loop via Hydra |
| `make train-evidence` | Launch the span extractor loop |
| `make infer`      | Score candidates and emit ranked predictions |
| `make eval`       | Evaluate the ranker on the validation split |
| `make mlflow-up`  | Serve the local MLflow UI against `sqlite:///mlflow.db` |
| `make quickstart` | `make demo-data` + `make train-criteria` |

## MLflow Tracking

Every entrypoint:

1. Calls `mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)` (SQLite by default).
2. Sets the experiment `gemini_reranker` and run name `hydra.job.name` unless overridden.
3. Logs the resolved Hydra config, Git metadata, hardware snapshot, metrics, and (optionally) checkpoints.

Set `MLFLOW_TRACKING_URI` externally or override `mlflow.tracking_uri` to direct runs elsewhere. To skip checkpoint uploads in constrained environments, run with `train.save_top_k=0 train.save_every_steps=0`.

## Testing & CI

```bash
make test          # pytest -q
make lint          # ruff check .
make type          # mypy src
```

`.github/workflows/ci.yml` executes formatting, linting, typing, demo data generation, and the test suite on each push/PR. A `.pre-commit-config.yaml` is provided for local hooks (ruff, mypy, EOF checks).

## Judge Providers

The default judge config (`conf/judge/offline_mock.yaml`) uses a deterministic mock so the demo pipeline stays offline. Override `judge.provider=gemini` and set `GEMINI_API_KEY` when you are ready to call the real Gemini API.

## License

MIT. See `baselines/` README for provenance of the mirrored Optuna checkpoint and augmentation files.
