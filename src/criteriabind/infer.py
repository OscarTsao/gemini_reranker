"""Legacy inference entrypoint bridging to Hydra CLI."""

from __future__ import annotations

from .cli.infer import main


if __name__ == "__main__":
    main()
