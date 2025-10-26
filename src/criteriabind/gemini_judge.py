"""Legacy Gemini judge entrypoint bridging to Hydra CLI."""

from __future__ import annotations

from .cli.judge import main


if __name__ == "__main__":
    main()
