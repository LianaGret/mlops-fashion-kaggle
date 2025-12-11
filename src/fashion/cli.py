from __future__ import annotations

import sys

import fire

from fashion.commands.datasets import Datasets
from fashion.commands.models import Models
from fashion.config import load_env
from fashion.console import console, print_error, print_header


class FashionCTL:
    def __init__(self) -> None:
        self.datasets = Datasets()
        self.models = Models()

    def version(self) -> None:
        from fashion import __version__

        print_header("fashionctl", f"version {__version__}")


def main() -> None:
    load_env()
    try:
        fire.Fire(FashionCTL, name="fashionctl")
    except KeyboardInterrupt:
        console.print("\n[muted]Interrupted by user[/muted]")
        sys.exit(130)
    except Exception as error:
        print_error(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
