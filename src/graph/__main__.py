"""Entrypoint for `python -m src.graph` execution."""

import sys
from src.graph.cli import main

if __name__ == "__main__":
    sys.exit(main())
