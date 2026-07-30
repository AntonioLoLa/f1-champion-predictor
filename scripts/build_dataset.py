#!/usr/bin/env python
"""Build the model-ready feature matrix.

    python scripts/build_dataset.py

Reads curated data from the database, runs the feature pipeline, validates the schema
contract and writes a versioned Parquet file. Thin wrapper over ``f1cp dataset build`` so
the pipeline stage has exactly one implementation.
"""

from __future__ import annotations

import sys

from f1cp.cli import app


def main() -> None:
    """Run the dataset build command with the arguments given on the command line."""
    app(args=["dataset", "build", *sys.argv[1:]], prog_name="build_dataset.py")


if __name__ == "__main__":
    main()
