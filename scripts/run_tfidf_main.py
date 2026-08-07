from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.runner import run_tfidf_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run leakage-resistant TF-IDF analysis v2"
    )
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    manifest = run_tfidf_analysis(args.output, n_permutations=args.permutations)
    print(json.dumps({"status": manifest["status"], "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
