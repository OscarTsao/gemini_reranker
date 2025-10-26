PYTHON ?= python

.PHONY: setup env-check fmt lint type test mlflow-up demo-data judge pairs train-criteria train-evidence infer eval quickstart

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m nltk.downloader punkt

env-check:
	$(PYTHON) -c "import torch, platform; print(torch.__version__, torch.cuda.is_available(), platform.platform())"

fmt:
	ruff check --fix src tests

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest -q

mlflow-up:
	@MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1 --port 5000

demo-data:
	$(PYTHON) -m criteriabind.data.prepare_redsm5_data

judge:
	$(PYTHON) -m criteriabind.cli.judge

pairs:
	$(PYTHON) -m criteriabind.cli.pair_builder

train-criteria:
	$(PYTHON) -m criteriabind.train.train_criteria_ranker

train-evidence:
	$(PYTHON) -m criteriabind.train.train_evidence_span

infer:
	$(PYTHON) -m criteriabind.cli.infer

eval:
	$(PYTHON) -m criteriabind.cli.evaluate

quickstart:
	make demo-data && \
	make train-criteria
