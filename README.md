# Gemini Reranker

Hydra-driven pipelines for preference-based reranking and span extraction using Gemini-provided labels and MLflow tracking. The repository ships with an offline baseline checkpoint, deterministic demo data, and tooling to run the full candidate generation ➜ judging ➜ dataset ➜ training ➜ inference workflow on CPU in minutes.

## Environment Setup

Activate the environment with either Conda or a plain virtualenv before running the pipeline.

**Conda**

```bash
conda env create -f environment.yml
conda activate gemini-reranker
python -m nltk.downloader punkt
```

**Pip / virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows use: .venv\Scripts\activate
make setup                     # installs -e .[dev] (same as pip install -r requirements.txt) and downloads NLTK punkt
```

## Quickstart

With the environment active you can build the offline pipeline end-to-end in minutes:

```bash
make demo-data

# Candidate generation (top-8 sentences per criterion)
python -m criteriabind.cli.candidate_gen split=train candidate_gen.k=8

# Deterministic mock judge (no networking required)
python -m criteriabind.cli.judge split=train judge.provider=mock

# Convert judgments into pairwise/listwise datasets
python -m criteriabind.cli.pair_builder split=train

# Train the cross-encoder ranker offline
python -m criteriabind.train.train_criteria_ranker \
  train.max_steps=300 hydra.job.name=ranker_offline_demo

# Run offline inference on the held-out split
python -m criteriabind.cli.infer split=test
```

The commands above log into `mlflow.db` (SQLite) and emit artefacts under `.pytest_cache`/`outputs/…` when `output_dir` is overridden, or under `outputs/…` by default. To bring up a local MLflow UI:

```bash
make mlflow-up    # serves http://127.0.0.1:5000 with ./mlruns/ artefacts
```

## Why K? Choosing K

`candidate_gen.k` controls how many snippets per (note, criterion) move forward to judging and training. Larger values improve oracle recall (more chances to include the ground-truth evidence) but require more judging budget and increase training time. The default `k=8` balances recall with speed on the demo set. Keep an eye on the `recall@K` number printed by the generator—when it plateaus near 1.0, increasing `k` mostly adds noise. Override the value inline when you need deeper pools:

```bash
python -m criteriabind.cli.candidate_gen split=train candidate_gen.k=12
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

## Model Initialisation

The default configuration fine-tunes `microsoft/deberta-v3-small` from Hugging Face for both ranking and span extraction. If you have a previous checkpoint you want to warm-start from, set `model.from_pretrained_path=/path/to/checkpoint` (the directory should contain the usual `config.json`/`pytorch_model.bin` layout or a saved `training_state.pt`). Tokenisation is cached under `~/.cache/huggingface`.

## Tooling & Layout

- `conf/` – Hydra configuration tree (defaults, hardware, MLflow, data, models, judges).
- `src/criteriabind/` – package code (data loaders, models, training, CLI, utilities).
- `demo_data/` – synthetic ReDSM5-style JSONL splits created by `make demo-data`.
- `baselines/` – frozen baseline checkpoint and augmentation artefacts.
- `tests/` – pytest suite covering config composition, dataset caching, judge fallbacks, and offline end-to-end smoke runs.
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
| `make quickstart` | Alias for the offline demo pipeline (candidate → judge → pairs → train) |

## MLflow Tracking

Every entrypoint:

1. Calls `mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)` (SQLite by default).
2. Sets the experiment `gemini_reranker` and run name `hydra.job.name` unless overridden.
3. Logs the resolved Hydra config, Git metadata, hardware snapshot, metrics, and (optionally) checkpoints.

Set `MLFLOW_TRACKING_URI` externally or override `mlflow.tracking_uri` to direct runs elsewhere. To skip checkpoint uploads in constrained environments, run with `train.save_top_k=0 train.save_every_steps=0`.

## Privacy & Networking

The default configuration is entirely offline: candidate generation, judging, pair building, training, and inference run locally without network calls. Judging uses a deterministic mock provider whose decisions are reproducible under a fixed seed. When you are ready to use Gemini for labeling, set `judge.provider=gemini` and provide an API key via `GEMINI_API_KEY` or `judge.api_key`; the CLI will fail fast with a clear error if the key is missing. All artifacts stay on disk (`mlflow.db`, `./mlruns/`, `outputs/…`) so you can reason about privacy boundaries explicitly.

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
