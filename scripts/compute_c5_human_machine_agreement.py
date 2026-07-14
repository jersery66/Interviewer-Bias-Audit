#!/usr/bin/env python3
"""Compute blinded C5 agreement with a separate owner-only model crosswalk."""

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


def _read_coding_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute blinded C5 agreement from two completed rater sheets")
    parser.add_argument("--rater-a", type=Path, required=True)
    parser.add_argument("--rater-b", type=Path, required=True)
    parser.add_argument("--owner-crosswalk", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = compute_c5_human_machine_agreement(
        _read_coding_table(args.rater_a),
        _read_coding_table(args.rater_b),
        owner_crosswalk=_read_coding_table(args.owner_crosswalk),
        consensus=_read_coding_table(args.consensus) if args.consensus is not None else None,
    )
    atomic_write_csv(metrics, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
