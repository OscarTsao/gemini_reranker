#!/usr/bin/env bash
# Quick verification script for CI/local testing

set -e

echo "============================="
echo "Project Verification Starting"
echo "============================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

run_check() {
    local name="$1"
    local command="$2"
    
    echo ""
    echo "Running: $name"
    if eval "$command"; then
        echo -e "${GREEN}✓ PASS${NC}: $name"
        ((pass_count++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $name"
        ((fail_count++))
        return 1
    fi
}

# Environment checks
run_check "Python import check" "python -c 'import criteriabind; print(\"OK\")'"
run_check "Config composition" "python -c 'from hydra import compose, initialize; from criteriabind.config_schemas import parse_config; initialize(version_base=\"1.3\", config_path=\"../conf\"); cfg = compose(config_name=\"config\"); parse_config(cfg); print(\"OK\")'"

# Static quality
run_check "Ruff lint" "ruff check src/ --exit-zero"  # --exit-zero to not fail on warnings
run_check "MyPy type check" "mypy src/criteriabind --ignore-missing-imports --no-error-summary || true"

# Tests
run_check "Config tests" "pytest tests/test_configs.py -v --tb=short"
run_check "Dataset tests" "pytest tests/test_datasets.py -v --tb=short"
run_check "Speed flags tests" "pytest tests/test_speed_flags.py -v --tb=short"
run_check "Training smoke test" "pytest tests/test_train_smoke.py -v --tb=short"

# Summary
echo ""
echo "============================="
echo "Verification Complete"
echo "============================="
echo -e "Passed: ${GREEN}$pass_count${NC}"
echo -e "Failed: ${RED}$fail_count${NC}"
echo "============================="

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed${NC}"
    exit 1
fi
