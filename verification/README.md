# Verification System for Gemini Reranker

This directory contains the comprehensive verification system for the `gemini_reranker` project.

## Quick Start

### Run Verification

```bash
# Quick check (< 10 seconds)
pytest tests/ -v --tb=short

# Full verification with reports
./scripts/run_verification.sh
```

### View Results

```bash
# Human-readable report
cat verification/report.md

# Machine-readable report
cat verification/report.json

# Detailed plan
cat verification/VERIFICATION_PLAN.md
```

## What Gets Verified

### 1. Environment & Dependencies (✅ PASS)
- Python 3.10+ installed
- All dependencies from `pyproject.toml` available
- Package imports correctly
- NLTK data downloaded

### 2. Configuration System (✅ PASS)
- Hydra configs compose successfully
- Overrides work correctly
- MLflow tracking URI configured
- Hardware settings resolved

### 3. MLflow Tracking (✅ PASS)
- SQLite backend (`mlflow.db`) created
- Local artifact store (`mlruns/`) functional
- Metrics and parameters logged
- Runs searchable

### 4. Training Pipeline (✅ PASS)
- Demo data generation works
- Model initialization successful
- Training runs without errors
- Checkpoints saved correctly
- MLflow tracking integration verified

### 5. Performance Optimization (✅ PASS)
- Device placement (CPU/CUDA) works
- Mixed precision (AMP) configured correctly
- torch.compile fallback graceful
- Gradient checkpointing available

### 6. Code Quality (✅ PASS)
- Linting passes (Ruff)
- Type checking clean (MyPy)
- All tests passing (Pytest)

## Directory Structure

```
verification/
├── README.md                  # This file
├── VERIFICATION_PLAN.md       # Detailed verification strategy
├── report.md                  # Human-readable verification report
├── report.json                # Machine-readable verification report
├── logs/                      # Execution logs
└── artifacts/                 # Test artifacts and outputs
```

## Configuration Files

### `conf/verification.yaml`

Verification-specific configuration for fast CI runs:

```yaml
seed: 42
train:
  max_steps: 300
  batch_size: 8
data:
  max_samples: 200
thresholds:
  min_metric_value: 0.0
  max_metric_tolerance: 1e-6
```

## Test Suite

### Core Tests

Located in `tests/`:

1. **`test_configs.py`** - Hydra configuration composition
2. **`test_datasets.py`** - Dataset loading and processing
3. **`test_speed_flags.py`** - Hardware optimization flags
4. **`test_train_smoke.py`** - End-to-end training smoke test

### Running Individual Tests

```bash
# Config tests
pytest tests/test_configs.py -v

# Dataset tests
pytest tests/test_datasets.py -v

# Speed flags tests
pytest tests/test_speed_flags.py -v

# Training smoke test
pytest tests/test_train_smoke.py -v
```

## Verification Scripts

### `scripts/run_verification.sh`

Quick verification script for CI/local testing:

```bash
./scripts/run_verification.sh
```

**Features**:
- Colored output (green ✓ for pass, red ✗ for fail)
- Summary of passed/failed checks
- Exit code 0 for success, 1 for failure

## Latest Verification Results

**Date**: 2025-10-26T04:00:00Z  
**Status**: ✅ **PASS**  
**Duration**: 8.53 seconds  
**Tests**: 5 passed, 0 failed

### Summary

- ✅ Package imports correctly
- ✅ Hydra configuration working
- ✅ MLflow local backend functional
- ✅ Training pipeline verified end-to-end
- ✅ Hardware optimization fallbacks graceful
- ✅ All tests passing

### Environment

- Python: 3.13.3
- PyTorch: 2.6.0.dev20250224+cu126
- CUDA: Available but incompatible GPU (correctly falls back to CPU)
- OS: Linux 6.8.0-85-generic

## CI Integration

### GitHub Actions Example

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

## Troubleshooting

### GPU Compatibility

**Issue**: GPU present but not compatible with PyTorch version

**Expected Behavior**: Tests correctly fall back to CPU

**Solution**: No action needed - this is expected and handled gracefully

### Warnings in Tests

**Issue**: 115 warnings during test execution

**Analysis**:
- Pydantic deprecation (MLflow): External, non-critical
- CUDA compatibility: Expected due to incompatible GPU
- Tensor pin_memory: PyTorch deprecation, non-critical
- Multiprocessing fork: Expected in test environment

**Action**: None required - all warnings are non-critical

## Maintenance

### Adding New Tests

When adding features to the codebase:

1. Add unit tests in `tests/`
2. Update smoke test if pipeline changes
3. Document new config options in `CLAUDE.md`
4. Run verification suite: `pytest tests/ -v`

### Updating Dependencies

When updating dependencies:

1. Run full verification: `./scripts/run_verification.sh`
2. Check for deprecated APIs
3. Update pinned versions in `requirements.txt`
4. Test on both CPU and GPU environments (if applicable)

## References

- **Verification Plan**: `VERIFICATION_PLAN.md`
- **Latest Report**: `report.md` (Markdown) or `report.json` (JSON)
- **Config Documentation**: `../CLAUDE.md`
- **Tests**: `../tests/`
- **CI Workflow**: `../.github/workflows/ci.yml`

## Success Criteria

Verification is successful if:

1. ✅ All tests in `tests/` pass
2. ✅ No critical lint/type errors
3. ✅ Smoke training completes without errors
4. ✅ MLflow tracking works (SQLite + mlruns)
5. ✅ Hydra configs compose correctly
6. ✅ Speed flags apply/fallback gracefully
7. ✅ CLI entrypoints respond to `--help`

## Support

For issues or questions about verification:

1. Check `VERIFICATION_PLAN.md` for detailed strategy
2. Review `report.md` for latest results
3. Run individual tests to isolate issues
4. Consult `CLAUDE.md` for project-specific guidance

---

**Last Updated**: 2025-10-26  
**Status**: ✅ All verification checks passing
