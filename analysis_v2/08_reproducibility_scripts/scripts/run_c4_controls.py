from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.c4_runner import run_c4_controls


DEFAULT_TURNS = Path("analysis_v2/01_inputs/c4_turns_official142_frozen.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C4 interviewer mechanism controls")
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--turns", type=Path, default=DEFAULT_TURNS)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    manifest = run_c4_controls(
        args.output,
        turns_path=args.turns,
        n_permutations=args.permutations,
    )
    print(json.dumps({"status": manifest["status"], "tests": manifest["family_test_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
