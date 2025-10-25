PYTHON ?= python

.PHONY: setup test train-criteria train-evidence judge infer lint format typecheck sec coverage ci verify candidate-gen pair-build clean prepare-demo prepare-redsm5

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]
	$(PYTHON) -m nltk.downloader punkt

test:
	$(PYTHON) -m pytest

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/ scripts/

sec:
	bandit -q -r src/ -ll

coverage:
	$(PYTHON) -m pytest --cov=criteriabind --cov-report=html --cov-report=term

ci: lint typecheck test sec

verify:
	$(PYTHON) scripts/verify_build.py

prepare-demo:
	$(PYTHON) scripts/prepare_demo_data.py

prepare-redsm5:
	$(PYTHON) scripts/prepare_redsm5_data.py

candidate-gen:
	$(PYTHON) -m criteriabind.candidate_gen --in data/raw/redsm5_train.jsonl --out data/proc/redsm5_judging_jobs.jsonl --k 8

judge-mock:
	$(PYTHON) -m criteriabind.gemini_judge --in-path data/proc/redsm5_judging_jobs.jsonl --out-path data/judged/redsm5_train.jsonl --mock

pair-build:
	$(PYTHON) -m criteriabind.pair_builder --in data/judged/redsm5_train.jsonl --out-train data/pairs/redsm5_criteria_train.jsonl --out-dev data/pairs/redsm5_criteria_dev.jsonl --out-test data/pairs/redsm5_criteria_test.jsonl

train-criteria:
	$(PYTHON) -m criteriabind.train_criteria_ranker --config configs/criteria_train.yaml

train-evidence:
	$(PYTHON) -m criteriabind.train_evidence_span --config configs/evidence_train.yaml

judge:
	$(PYTHON) -m criteriabind.gemini_judge --in-path data/proc/redsm5_judging_jobs.jsonl --out-path data/judged/redsm5_train.jsonl

infer:
	$(PYTHON) -m criteriabind.infer --notes-path data/raw/redsm5_test.jsonl --output-path data/proc/redsm5_inference.jsonl

clean:
	rm -rf data/proc/* data/judged/* data/pairs/* data/models/* artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/ .mypy_cache/ .ruff_cache/
