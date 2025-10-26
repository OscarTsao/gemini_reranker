"""Convert the ReDSM5 CSV release into JSONL splits for Criteria Bind."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from criteriabind.datasets import prepare_redsm5_splits
from criteriabind.logging_utils import get_logger


LOGGER = get_logger(__name__)


@dataclass
class PrepareArgs:
    """CLI arguments for preparing the ReDSM5 dataset."""

    data_dir: Path = Path("data/redsm5")
    output_dir: Path = Path("data/raw")
    dev_ratio: float = 0.15
    test_ratio: float = 0.15
    include_special_case: bool = True
    prefix: str = "redsm5"


def main(args: PrepareArgs) -> None:
    paths = prepare_redsm5_splits(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        include_special_case=args.include_special_case,
        prefix=args.prefix,
    )
    summary = ", ".join(f"{split}: {path.name}" for split, path in sorted(paths.items()))
    LOGGER.info("Prepared ReDSM5 JSONL splits (%s)", summary)


if __name__ == "__main__":
    tyro.cli(main)
