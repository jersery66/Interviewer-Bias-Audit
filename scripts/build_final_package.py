from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reanalysis_v2.reporting import build_final_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tables, figures, validation reports, and hash audits for reanalysis v2."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ROOT / "analysis_v2",
        help="Path to the frozen analysis_v2 directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <analysis-root>/06_tables_figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_final_package(args.analysis_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
