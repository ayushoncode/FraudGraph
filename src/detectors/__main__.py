"""Entrypoint for `python -m src.detectors` execution."""

import sys
from src.detectors.cli import main

if __name__ == "__main__":
    sys.exit(main())
