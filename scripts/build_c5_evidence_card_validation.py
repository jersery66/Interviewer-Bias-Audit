"""Build the restricted C5 evidence-card package without creating workbooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reanalysis_v2.c5_evidence_card_validation import (
    C5_CANONICAL_RELATIVE,
    PARTICIPANT_TEXT_RELATIVE,
    build_evidence_card_package,
)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="原子化准备受限的 C5 证据卡 JSON 与审计清单。"
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    canonical_path = project_root / C5_CANONICAL_RELATIVE
    participant_text_path = project_root / PARTICIPANT_TEXT_RELATIVE

    try:
        paths = build_evidence_card_package(
            canonical_path,
            participant_text_path,
            args.output_root,
            seed=args.seed,
        )
    except Exception as exc:
        parser.exit(
            1,
            "error: C5 evidence-card package preparation failed "
            f"({type(exc).__name__})\n",
        )

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    summary = {
        "status": manifest["status"],
        "output_root": str(paths.root),
        "manifest_path": str(paths.manifest_json),
        "training_total": manifest["training_total"],
        "formal_total": manifest["formal_total"],
        "canonical_sha256": manifest["canonical_sha256"],
        "participant_text_sha256": manifest["participant_text_sha256"],
        "reviewer_material_release_status": manifest[
            "reviewer_material_release_status"
        ],
        "file_hashes": manifest["file_hashes"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
