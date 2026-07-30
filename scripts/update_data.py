#!/usr/bin/env python
"""Acquire Formula 1 data.

First run:
    python scripts/update_data.py --bootstrap

Subsequent runs (roughly seven API requests per new round):
    python scripts/update_data.py

This is a thin wrapper over ``f1cp data update`` so the pipeline stage has exactly one
implementation.
"""

from __future__ import annotations

import sys

from f1cp.cli import app


def main() -> None:
    """Run the data update command with the arguments given on the command line."""
    app(args=["data", "update", *sys.argv[1:]], prog_name="update_data.py")


if __name__ == "__main__":
    main()
