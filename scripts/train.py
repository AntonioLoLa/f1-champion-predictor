#!/usr/bin/env python
"""Train the model zoo.

    python scripts/train.py

Walks forward over completed seasons, fits, calibrates and scores every enabled model-zoo
candidate, selects a winner and activates it. Thin wrapper over ``f1cp train run`` so the
pipeline stage has exactly one implementation.
"""

from __future__ import annotations

import sys

from f1cp.cli import app


def main() -> None:
    """Run the training command with the arguments given on the command line."""
    app(args=["train", "run", *sys.argv[1:]], prog_name="train.py")


if __name__ == "__main__":
    main()
