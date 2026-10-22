"""Entrypoint allowing `python -m src.generator` execution."""

import sys
from src.generator.cli import main

if __name__ == "__main__":
    sys.exit(main())
