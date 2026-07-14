#!/usr/bin/env python3
"""Compute C5 human-machine and inter-rater agreement after manual coding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.interviewer_signal_revision import compute_c5_human_machine_agreement
from reanalysis_v2.io import atomic_write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute C5 agreement from two completed rater sheets")
    parser.add_argument("--rater-a", type=Path, required=True)
    parser.add_argument("--rater-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = compute_c5_human_machine_agreement(
        pd.read_csv(args.rater_a),
        pd.read_csv(args.rater_b),
    )
    atomic_write_csv(metrics, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
