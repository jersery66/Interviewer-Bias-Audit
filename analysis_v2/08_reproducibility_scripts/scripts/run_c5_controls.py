from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.c5_runner import run_c5_controls


DEFAULT_SPANS = Path(
    r"F:\数据库\DAIC-WOZ\processed_research\verified_source_data_official142_20260703"
    r"\03_canonical_text_official142\reviewed_c5_spans_official142.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C5 evidence concentration controls")
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    manifest = run_c5_controls(
        args.output,
        spans_path=args.spans,
        n_permutations=args.permutations,
    )
    print(json.dumps({"status": manifest["status"], "tests": manifest["family_test_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
