# Gemini Reranker - End-to-End Verification Plan

## Executive Summary

This document outlines the comprehensive verification strategy for the `gemini_reranker` project, ensuring correctness, reproducibility, and production-readiness.

## Verification Objectives

1. **Build & Installation**: Verify clean installation in isolated environment
2. **Configuration System**: Confirm Hydra as single source of truth for parameters
3. **MLflow Tracking**: Verify SQLite backend (`./mlflow.db`) and local artifact store (`./mlruns/`)
4. **Offline End-to-End**: Execute complete pipelines without external API calls
5. **Determinism**: Confirm seed-based reproducibility
6. **Performance**: Validate optimization flags (AMP, compile, checkpointing)
7. **Quality Gates**: Run linting, type-checking, and tests
8. **CLI/Build System**: Verify Makefile targets and CLI entrypoints

## Verification Components

### 1. Environment & Dependencies

**Files**:
- `tests/test_configs.py` - Hydra configuration composition
- `tests/test_speed_flags.py` - Hardware optimization flags

**Checks**:
- ✅ Python 3.10+ installed
- ✅ All dependencies from `pyproject.toml` available
- ✅ Package imports correctly (`import criteriabind`)
- ✅ Hydra configs compose successfully
- ✅ MLflow tracking backend accessible

**Command**: 
```bash
python -c "import criteriabind; print('OK')"
pytest tests/test_configs.py -v
```

### 2. Configuration System (Hydra)

**Files**:
- `conf/config.yaml` - Base configuration
- `conf/verification.yaml` - Verification-specific overrides
- `conf/mlflow/local.yaml` - MLflow local backend config
- `conf/hardware/auto.yaml` - Hardware settings
- `conf/train/ranker_fast.yaml` - Fast training config
- `conf/judge/offline_mock.yaml` - Mock judge for offline testing

**Checks**:
- ✅ Base config composes without errors
- ✅ Overrides apply correctly (`hardware.device=cpu`, `train.max_steps=30`)
- ✅ MLflow tracking URI resolves to `sqlite:///mlflow.db`
- ✅ Artifact root points to `./mlruns/`
- ✅ Hydra output dir follows pattern `outputs/YYYY-MM-DD/HH-MM-SS-<job>`

**Command**:
```bash
pytest tests/test_configs.py -v
```

### 3. MLflow Local Backend

**Verification Points**:
- SQLite database file (`mlflow.db`) created
- Local artifact directory (`mlruns/`) exists
- Metrics logged successfully
- Parameters logged successfully
- Artifacts (checkpoints, plots) stored correctly
- Runs searchable via MLflowClient

**Test**: `tests/test_train_smoke.py`

**Expected Behavior**:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("gemini_reranker")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])
assert len(runs) > 0
```

### 4. Data Pipeline

**Components**:
- ✅ Demo data generation (`scripts/prepare_demo_data.py`)
- ✅ Candidate generation
- ✅ Mock Gemini judge (`scripts/mock_gemini.py`)
- ✅ Pair building from judged items

**Test**: `tests/test_datasets.py`

**Checks**:
- Demo data JSONL files created
- Candidates extracted from notes
- Mock judge produces deterministic rankings
- Pairwise training rows generated correctly

### 5. Training Pipelines

#### Track A: Criteria Ranker

**Test**: `tests/test_train_smoke.py`

**Pipeline**:
1. Load pairwise dataset
2. Initialize `CrossEncoderRanker` model
3. Train for minimal steps (smoke test)
4. Log metrics to MLflow
5. Save checkpoint
6. Verify checkpoint loadable

**Validation Checks**:
- ✅ Training loss decreases (or stays finite)
- ✅ Metrics logged to MLflow (e.g., `train/loss`, `val/map@10`)
- ✅ Checkpoint saved to correct path
- ✅ Hydra output directory created
- ✅ Device placement correct (CPU/CUDA as configured)
- ✅ AMP dtype applied correctly

**Config Overrides**:
```yaml
hardware.device=cpu
hardware.compile=false
train.max_steps=1
train.eval_every_steps=1
data.max_samples=4
```

#### Track B: Evidence Span

**Pipeline**: *(Similar to Track A but for span extraction)*

1. Load QA-style span dataset
2. Initialize `QASpanModel`
3. Train for minimal steps
4. Log metrics (`val/f1_at_iou`)
5. Save checkpoint

### 6. Performance Optimization Flags

**Test**: `tests/test_speed_flags.py`

**Checks**:
- ✅ `hardware.device=auto` resolves to CPU or CUDA
- ✅ `hardware.compile=true` applies torch.compile on PyTorch 2.0+
- ✅ `hardware.compile=false` fallback works
- ✅ `hardware.amp_dtype=bf16` applies mixed precision on supported hardware
- ✅ `hardware.amp_dtype=fp16` fallback on older GPUs
- ✅ `hardware.amp_dtype=float32` on CPU
- ✅ `hardware.gradient_checkpointing=true` enables checkpointing
- ✅ TF32 enabled on Ampere+ GPUs
- ✅ Fused AdamW used when available

**Fallback Behavior**:
- No GPU → `compile` disabled, `amp_dtype=float32`
- PyTorch < 2.0 → `compile` disabled
- No bf16 support → fallback to fp16 or fp32
- All fallbacks should not crash, only log warnings

### 7. Determinism & Reproducibility

**Checks**:
- ✅ `seed=42` set in config
- ✅ Same seed produces identical results (within tolerance)
- ✅ Checkpoint hash identical across runs with same seed
- ✅ Split assignment deterministic (based on note ID hash)

**Test**:
```python
# Run training twice with same seed
run1_metric = train(seed=42)
run2_metric = train(seed=42)
assert abs(run1_metric - run2_metric) < 1e-6
```

### 8. Quality Gates

#### Linting (Ruff)
```bash
ruff check src/ tests/ scripts/
```

**Expected**: No critical errors

#### Type Checking (MyPy)
```bash
mypy src/criteriabind --ignore-missing-imports
```

**Expected**: No type errors in core modules

#### Tests (Pytest)
```bash
pytest tests/ -v --tb=short
```

**Expected**: All tests pass

**Current Tests**:
- `test_configs.py` - Configuration composition
- `test_datasets.py` - Dataset loading and processing
- `test_speed_flags.py` - Hardware optimization flags
- `test_train_smoke.py` - End-to-end training smoke test

### 9. CLI & Makefile

**Makefile Targets**:
- `make demo-data` - Generate demo dataset
- `make train-criteria` - Train criteria ranker
- `make train-evidence` - Train evidence span model
- `make test` - Run pytest suite
- `make lint` - Run ruff check

**CLI Entrypoints**:
- `python -m criteriabind.cli.candidate_gen --help`
- `python -m criteriabind.cli.judge --help`
- `python -m criteriabind.cli.pair_builder --help`
- `python -m criteriabind.train.train_criteria_ranker --help`
- `python -m criteriabind.train.train_evidence_span --help`
- `python -m criteriabind.cli.infer --help`

**Checks**:
- ✅ All entrypoints show help text
- ✅ Tyro argument parsing works
- ✅ Required arguments enforced
- ✅ Invalid arguments produce clear error messages

## Verification Workflow

### Quick Verification (< 2 minutes)

```bash
# 1. Environment
python -c "import criteriabind; print('✓ Imports OK')"

# 2. Config
pytest tests/test_configs.py -v

# 3. Speed flags
pytest tests/test_speed_flags.py -v

# 4. Datasets
pytest tests/test_datasets.py -v
```

### Full Verification (< 5 minutes)

```bash
# Run all tests including training smoke test
pytest tests/ -v --tb=short

# Check linting
ruff check src/ tests/ scripts/

# Verify MLflow backend
ls mlflow.db mlruns/
```

### CI Verification

**GitHub Actions**:
```yaml
name: Verification
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short
      - run: ruff check src/ tests/ scripts/
```

## Verification Report Schema

After running verification, generate reports:

### JSON Report (`verification/report.json`)

```json
{
  "timestamp": "2025-10-26T...",
  "verdict": "PASS" | "FAIL",
  "summary": {
    "total_checks": 10,
    "passed": 9,
    "failed": 1,
    "duration_seconds": 120.5
  },
  "env": {
    "python": "3.10.12",
    "torch": "2.1.0",
    "cuda": true,
    "gpus": ["NVIDIA GeForce RTX 3090"]
  },
  "results": [
    {
      "check_id": "E1",
      "name": "Package imports",
      "status": "PASS",
      "duration_seconds": 0.5
    },
    ...
  ]
}
```

### Markdown Report (`verification/report.md`)

Human-readable summary with:
- Executive summary
- Environment snapshot
- Test results by category
- Performance metrics
- Recommendations

## Success Criteria

Verification is successful if:

1. ✅ All tests in `tests/` pass
2. ✅ No critical lint/type errors
3. ✅ Smoke training completes without errors
4. ✅ MLflow tracking works (SQLite + mlruns)
5. ✅ Hydra configs compose correctly
6. ✅ Speed flags apply/fallback gracefully
7. ✅ CLI entrypoints respond to `--help`

## Edge Cases & Fallbacks

| Scenario | Expected Behavior |
|----------|------------------|
| No GPU | `device=cpu`, `compile=false`, `amp_dtype=float32` |
| PyTorch < 2.0 | `compile=false`, warning logged |
| No bf16 support | Fallback to `fp16` or `fp32` |
| Missing NLTK data | Auto-download on first use |
| MLflow server down | SQLite local backend still works |

## Maintenance

### Adding New Features

When adding features:
1. Add unit tests in `tests/`
2. Update smoke test if pipeline changes
3. Document new config options in CLAUDE.md
4. Update verification plan if new dependencies added

### Updating Dependencies

When updating deps:
1. Run full verification suite
2. Check for deprecated APIs
3. Update pinned versions in `requirements.txt`
4. Test on both CPU and GPU environments

## References

- **Config Docs**: `CLAUDE.md`
- **Tests**: `tests/`
- **CI Workflow**: `.github/workflows/ci.yml`
- **Hydra Configs**: `conf/`

---

**Last Updated**: 2025-10-26
**Status**: ✅ Verification system implemented and tested
