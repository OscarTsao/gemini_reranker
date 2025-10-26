"""Legacy entrypoint bridging to the Hydra pair builder."""

from __future__ import annotations

from .cli.pair_builder import main


if __name__ == "__main__":
    main()
