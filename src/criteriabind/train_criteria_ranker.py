"""Legacy entrypoint redirecting to the Hydra-based trainer."""

from __future__ import annotations

from .train.train_criteria_ranker import main


if __name__ == "__main__":
    main()
