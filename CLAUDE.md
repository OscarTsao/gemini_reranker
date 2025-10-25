# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup & Development
```bash
make setup                    # Install dependencies with pip, download NLTK punkt
python scripts/prepare_demo_data.py  # Generate synthetic training data
make test                     # Run pytest suite
```

### Training & Inference Pipeline
```bash
make judge                    # Call Gemini API to judge candidates (requires GEMINI_API_KEY)
make train-criteria           # Train criteria ranker (Track A) using configs/criteria_train.yaml
make train-evidence           # Train evidence span model (Track B) using configs/evidence_train.yaml
make infer                    # Run inference on test data
```

### Direct CLI Usage
All modules support Tyro-powered CLI with flag overrides:
```bash
python -m criteriabind.candidate_gen --in-path data/raw/redsm5_train.jsonl --out-path data/proc/redsm5_judging_jobs.jsonl --k 8
python -m criteriabind.gemini_judge --in-path data/proc/redsm5_judging_jobs.jsonl --out-path data/judged/redsm5_train.jsonl --model gemini-2.5-flash
python -m criteriabind.pair_builder --in data/judged/redsm5_train.jsonl --out-train data/pairs/redsm5_criteria_train.jsonl --out-dev data/pairs/redsm5_criteria_dev.jsonl --out-test data/pairs/redsm5_criteria_test.jsonl
python -m criteriabind.train_criteria_ranker --pairs-path data/pairs/redsm5_criteria_train.jsonl --dev-path data/pairs/redsm5_criteria_dev.jsonl --model-name-or-path baselines/dataaug_trial_0043/model/best
```

### Running Single Test
```bash
pytest tests/test_schemas.py              # Test specific file
pytest tests/test_pair_builder.py::test_assign_split  # Test specific function
pytest -k "text_utils"                    # Test matching pattern
```

## Architecture

### Core Pipeline Flow
The repository implements a **preference learning pipeline** for clinical criteria matching:

1. **Candidate Generation** (`candidate_gen.py`): Extracts candidate text spans from clinical notes for each criterion
2. **Gemini Judging** (`gemini_judge.py`): Submits candidates to Gemini API with clinical rubric, enforces two-pass consistency, drops unsafe responses
3. **Pair Building** (`pair_builder.py`): Converts judge rankings into pairwise training examples, splits by note hash for determinism
4. **Training** (`train_criteria_ranker.py`, `train_evidence_span.py`): Trains BERT-like models with RankNet/hinge losses or span margin losses
5. **Inference** (`infer.py`): Produces best-of-k decisions and evidence snippets

### Two Training Tracks
- **Track A (Criteria Matching)**: `train_criteria_ranker.py` trains a `CrossEncoderRanker` (BERT + scalar head) with pairwise ranking loss to classify whether criteria are met
- **Track B (Evidence Binding)**: `train_evidence_span.py` trains a `QASpanModel` (AutoModelForQuestionAnswering) with span margin loss to extract supporting evidence text

### Key Abstractions

**Schemas** (`schemas.py`):
- `Sample`: Input note + criteria list
- `Candidate`: Text span with optional start/end offsets
- `JudgingJob`: Submitted to Gemini with criterion + candidates
- `JudgedItem`: Job result with winner index and ranking
- `PairwiseRow`: Training row with positive/negative pairs

**Models** (`models.py`):
- `CrossEncoderRanker`: BERT encoder + dropout + linear head for scalar scores
- `QASpanModel`: Wrapper around `AutoModelForQuestionAnswering` with margin loss for hard negatives
- `pairwise_loss()`: Implements RankNet (softplus) or hinge loss

**Config System** (`config.py`):
- Uses Pydantic dataclasses for type-safe YAML loading
- `TrainingConfig`: hyperparams, paths, MLflow settings
- `JudgeConfig`: Gemini model, rubric version, safety settings
- `build_run_config()`: Merges YAML with CLI args

### Gemini Judge Details
The judge (`gemini_judge.py`) implements:
- **JSON Mode**: Enforces structured output with schema validation
- **Two-pass consistency**: Runs identical prompts twice with variant wording, tie-breaks on disagreement
- **Safety filtering**: Drops items with safety flags from Gemini
- **Retry logic**: Uses tenacity for exponential backoff on 429/5xx errors

Rubric instructs Gemini to select the "most faithful, directly supportive, complete, clear, and safe" snippet.

### Data Flow & Directory Structure
```
data/
  raw/           # Input samples (Sample schema)
  proc/          # Candidate generation output (JudgingJob schema)
  judged/        # Gemini judge results (JudgedItem schema)
  pairs/         # Pairwise datasets (PairwiseRow schema)
  models/        # Training checkpoints (best.ckpt, last.ckpt)
```

Split assignment uses SHA256 hash of `note_id` for reproducibility across runs.

### Training Loop Architecture
Both training entrypoints follow the same pattern:
1. Load config from YAML or CLI args (Tyro dataclasses)
2. Call `seed_everything()` for determinism
3. Build PyTorch DataLoader with custom collate function
4. Initialize AdamW + cosine schedule with warmup
5. Support mixed precision (fp16/bf16) via torch.cuda.amp
6. Gradient accumulation for large effective batch sizes
7. Log to MLflow when `--mlflow_run_name` is provided
8. Save `best.ckpt` based on dev metric, `last.ckpt` on completion

Checkpoint format: `{model_state, optimizer_state, scheduler_state, step}`

### Inference Strategy
`infer.py` performs best-of-k inference:
- Generate k candidates per criterion
- Score all candidates with trained ranker
- Predict positive label if top score exceeds threshold
- Optionally extract QA span using evidence model

## Environment Variables
- `GEMINI_API_KEY`: Required for `make judge` and `gemini_judge.py`

Store in `.env` file or export directly. Never commit `.env` to git.

## Testing Strategy
- `tests/conftest.py`: Shared pytest fixtures
- `tests/test_schemas.py`: Pydantic validation and serialization
- `tests/test_pair_builder.py`: Split assignment determinism
- `tests/test_text_utils.py`: Sentence splitting and span extraction

When adding new utilities, include round-trip tests and edge cases.

## Code Style
- Python 3.10+, 4-space indentation, line length ≤100 (enforced by Ruff)
- Type hints required on public functions
- Tyro CLI pattern: `@dataclass` args + `tyro.cli(main)`
- Logging via `logging_utils.get_logger(__name__)`

## Determinism
All training scripts call `seed.seed_everything(seed)` to ensure:
- Reproducible random splits
- Consistent model initialization
- Deterministic PyTorch operations

Use the same seed across runs for exact reproduction.
