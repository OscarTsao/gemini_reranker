# Repository Guidelines

## Project Structure & Module Organization
- `src/criteriabind/` hosts the Hydra-driven pipelines, dataloaders, and CLI entrypoints; mirror existing subpackages when adding new modules.
- `conf/` holds Hydra defaults and overrides grouped by domain (`data/`, `hardware/`, `mlflow/`); keep new configs small and compose them via `defaults` lists.
- `tests/` mirrors the package layout with pytest suites and smoke checks for training loops; keep fixtures close to their consumers.
- Generated artifacts belong in `outputs/`, `demo_data/`, and `baselines/`; keep other directories clean.

## Build, Test, and Development Commands
- `make setup` installs the editable package with dev extras and downloads NLTK punkt.
- `make demo-data` seeds deterministic JSONL splits used by quickstarts.
- `make train-criteria` / `make train-evidence` launch the ranker and span-training Hydra jobs; pass overrides like `hardware.device=cpu` as needed.
- `make lint`, `make fmt`, `make type`, and `make test` run Ruff, Ruff with fixes, mypy, and pytest; prefer these wrappers to direct tool invocations.

## Coding Style & Naming Conventions
- Python targets 3.10 with 4-space indentation and Ruff line length 100; keep imports sorted (`ruff check --fix` handles it).
- Name modules and configs with descriptive snake_case; Hydra config groups should stay lowercase to match existing patterns.
- When adding CLI entrypoints, expose them via `__main__` modules under `src/criteriabind/cli/` and add `Makefile` targets when they aid discoverability.

## Testing Guidelines
- Write pytest cases under `tests/` mirroring the package path (e.g., `tests/train/test_ranker.py` for `src/criteriabind/train/...`).
- Exercise new Hydra configs with smoke tests that run fast on CPU; gate long runs behind markers such as `@pytest.mark.slow`.
- Cover new branches with assertions rather than relying on golden logs; share fixtures via `tests/data/` instead of hardcoded paths.
- Run `make test` locally before pushing; call out any focused `pytest` runs in the PR description.

## Commit & Pull Request Guidelines
- Follow the repository’s short, imperative commit style (`Add`, `Update`, `Fix`); keep subjects under ~65 characters and link issues in the body if relevant.
- Squash noisy WIP commits before opening a PR; describe the change, note affected Hydra configs, and call out new artifacts or migrations.
- Provide verification notes: list `make` targets or ad-hoc commands you ran, add MLflow run IDs when applicable, and include UI screenshots when relevant.
- Request reviews after CI passes or you have explained expected failures; link dependent PRs or experiments so reviewers can reconstruct the pipeline.
