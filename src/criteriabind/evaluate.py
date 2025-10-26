"""Legacy evaluation entrypoint bridging to Hydra CLI."""

from __future__ import annotations

from .cli.evaluate import main


if __name__ == "__main__":
    main()
