# Repository Guidelines

## Project Structure & Module Organization
- `src/criteriabind/`: core package modules (configs, candidate generation, Gemini judge, training, evaluation, inference).
- `tests/`: unit tests for schemas, text utilities, and pair builders; add new suites alongside existing patterns.
- `scripts/`: helper utilities such as `prepare_demo_data.py` for smoke data.
- `data/`: expected workspace for raw inputs, Gemini judgments, pairwise datasets, and checkpoints; kept empty in git.
- `configs/`: YAML run configs consumed by Tyro CLIs and Make targets.

## Build, Test, and Development Commands
- `make setup` — install dependencies in editable mode and download NLTK punkt.
- `python scripts/prepare_demo_data.py` — generate a minimal example dataset under `data/raw/`.
- `make judge` — call Gemini to create preference labels (requires `GEMINI_API_KEY`).
- `make train-criteria` / `make train-evidence` — launch RankNet and QA training loops using configs.
- `make infer` — run best-of-k inference over `data/raw/test.jsonl`.
- `make test` — execute the pytest suite.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8 defaults with 4-space indentation; keep line length ≤100 (matches `tool.ruff`).
- Prefer descriptive module-level docstrings and concise comments when logic is non-obvious.
- Type hints are required on public functions; follow existing dataclass patterns.
- Use Tyro CLI patterns for new entrypoints (`@dataclass` args + `tyro.cli`).

## Testing Guidelines
- Framework: `pytest`; place tests under `tests/` mirroring module names (`test_<module>.py`).
- For new utilities, include round-trip and edge-case tests; mock external services when needed.
- Run `make test` locally before committing; add regression fixtures under `tests/data/` when appropriate.

## Commit & Pull Request Guidelines
- Write imperative commit messages (e.g., “Add pairwise ranker dataset builder”); group related changes.
- PRs should summarize problem + solution, list test commands executed, and link to relevant issues.
- Include configuration notes (e.g., environment variables, required data files) in PR descriptions when applicable.

## Security & Configuration Tips
- Store secrets such as `GEMINI_API_KEY` in `.env` or your secrets manager; never commit `.env`.
- Validate API usage against Google rate limits; the judge client already retries on 429/5xx but avoid unnecessary parallelism.
