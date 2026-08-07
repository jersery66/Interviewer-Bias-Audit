from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.prepare import freeze_c4_turns_input, prepare_primary_inputs


DEFAULT_SOURCE = Path(
    r"F:\数据库\DAIC-WOZ\processed_research\verified_source_data_official142_20260703"
    r"\03_canonical_text_official142\mpnet_conditions_official142_verified.csv"
)
DEFAULT_TURNS_SOURCE = Path(
    r"F:\数据库\DAIC-WOZ\processed_research\verified_source_data_official142_20260703"
    r"\03_canonical_text_official142\turns_official142.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze DAIC-WOZ analysis v2 inputs and splits")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--turns-source", type=Path, default=DEFAULT_TURNS_SOURCE)
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    args = parser.parse_args()
    manifest = prepare_primary_inputs(args.source, args.output)
    freeze_c4_turns_input(args.turns_source, args.output)
    print(json.dumps(manifest["cohort"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
