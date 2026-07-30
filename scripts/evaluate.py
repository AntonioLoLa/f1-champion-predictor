#!/usr/bin/env python
"""Evaluate a trained model.

    python scripts/evaluate.py

Scores a model's leakage-free out-of-fold predictions with the full metric suite, renders
Plotly figures, runs SHAP and permutation importance, and writes a report. Thin wrapper
over ``f1cp evaluate run`` so the pipeline stage has exactly one implementation.
"""

from __future__ import annotations

import sys

from f1cp.cli import app


def main() -> None:
    """Run the evaluation command with the arguments given on the command line."""
    app(args=["evaluate", "run", *sys.argv[1:]], prog_name="evaluate.py")


if __name__ == "__main__":
    main()
