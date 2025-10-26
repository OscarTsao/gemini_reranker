"""Legacy entrypoint redirecting to the Hydra-based span trainer."""

from __future__ import annotations

from .train.train_evidence_span import main


if __name__ == "__main__":
    main()
