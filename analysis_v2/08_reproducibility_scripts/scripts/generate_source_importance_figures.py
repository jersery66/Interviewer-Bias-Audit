from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.source_importance_figures import build_source_importance_figures


def main() -> int:
    analysis_root = REPO_ROOT / "analysis_v2"
    output_dir = analysis_root / "06_tables_figures" / "figures"
    result = build_source_importance_figures(analysis_root, output_dir)
    print(
        f"source_importance_figures_complete count={result['figure_count']} "
        f"output={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
