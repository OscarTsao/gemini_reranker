"""Legacy entrypoint bridging to the Hydra candidate generator."""

from __future__ import annotations

from .cli.candidate_gen import main


if __name__ == "__main__":
    main()
