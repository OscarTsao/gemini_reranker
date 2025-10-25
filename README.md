# Criteria Bind

Criteria Bind refines an SFT-tuned BERT-like model for DSM-style clinical criteria matching (Track A) and evidence binding (Track B) using Gemini judged preferences. The repository contains deterministic training scripts, MLflow logging, candidate generation, and evaluation tooling.

## Quickstart

```bash
make setup
python scripts/prepare_demo_data.py
python scripts/prepare_redsm5_data.py
make judge          # requires GEMINI_API_KEY
make train-criteria
make train-evidence
make infer
make test
```

The `Makefile` targets assume YAML configs in `configs/`. You can also run the CLIs directly with Tyro-powered flags, for example:

```bash
python -m criteriabind.candidate_gen --in-path data/raw/redsm5_train.jsonl --out-path data/proc/redsm5_judging_jobs.jsonl --k 8
python -m criteriabind.gemini_judge --in-path data/proc/redsm5_judging_jobs.jsonl --out-path data/judged/redsm5_train.jsonl --model gemini-2.5-flash
python -m criteriabind.pair_builder --in data/judged/redsm5_train.jsonl --out-train data/pairs/redsm5_criteria_train.jsonl --out-dev data/pairs/redsm5_criteria_dev.jsonl --out-test data/pairs/redsm5_criteria_test.jsonl
python -m criteriabind.train_criteria_ranker --pairs-path data/pairs/redsm5_criteria_train.jsonl --dev-path data/pairs/redsm5_criteria_dev.jsonl --model-name-or-path baselines/dataaug_trial_0043/model/best
```

## Baseline Artifacts

The Optuna trial `0043` DeBERTa classifier from `DataAugmentation_Evaluation` (test F1 **0.8535**) is mirrored under `baselines/dataaug_trial_0043/`. It includes the frozen checkpoint, validation/test metrics, and the hybrid/nlpaug/textattack augmentation CSVs plus ground truth labels. Use it as a warm-start checkpoint or to benchmark Gemini-judged tuning runs.

## Project Layout

- `src/criteriabind/` – core package modules (configs, schemas, candidate generation, models, training loops, evaluation, inference).
- `scripts/prepare_demo_data.py` – creates a tiny synthetic training file for smoke tests.
- `data/` – structured raw, judged, pairwise, and model output directories.
- `configs/` – example YAML configs for reproducible runs (add your own variants).
- `tests/` – unit tests for text utilities, schemas, and pair building.

## Environment

Set `GEMINI_API_KEY` in your environment or `.env` file to enable Gemini judging. Sample layout in `.env.example`.

## Determinism & Logging

All training entrypoints call `seed.seed_everything`, support gradient accumulation, mixed precision (`fp16`/`bf16`), resume checkpoints, and log metrics to MLflow when `--mlflow_run_name` is provided. Checkpoints are saved under `data/models/`.

## Judging Workflow

1. Generate candidate snippets (`candidate_gen.py`).
2. Call Gemini judge (`gemini_judge.py`) which enforces JSON mode, two-pass consistency, and safety checks.
3. Build pairwise/listwise datasets (`pair_builder.py`), split by note hash.

## Training Tracks

- **Track A (Criteria Matching)**: `train_criteria_ranker.py` trains a cross-encoder with RankNet or hinge losses, logging pairwise accuracy and saving `best.ckpt`.
- **Track B (Evidence Binding)**: `train_evidence_span.py` trains an extractive QA model with span margin loss for hard negatives.

## Inference & Evaluation

`infer.py` produces best-of-k decisions, returning labels and top evidence snippets. `evaluate.py` computes metrics for criteria classification and span accuracy for evidence binding.

## Testing

Unit tests cover schema validation, pair construction, and text utilities:

```bash
make test
```

Extend with additional tests for your data loaders or trainers as needed.
