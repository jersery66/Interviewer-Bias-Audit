"""Open the packet-3 C5 evidence round after human consensus is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.c5_blind_workflow_v3 import (
    build_stage2_evidence_forms,
    verify_consensus_freeze,
)
from scripts.prepare_c5_blind_audit_v3 import (
    DEFAULT_PRIVATE_PACKET,
    DEFAULT_PUBLIC_PACKET,
    DEFAULT_REVIEWED_SPANS_PATH,
)


REVIEWER_A_ORDER_SEED = 202607213
REVIEWER_B_ORDER_SEED = 202607214


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _shuffle(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(int(seed))
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def prepare_stage2(
    *,
    consensus_path: Path,
    freeze_record_path: Path,
    blind_key_path: Path,
    reviewed_spans_path: Path,
    line_index_path: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    # This check must occur before any C5 span is read or reviewer file written.
    freeze_record = verify_consensus_freeze(consensus_path, freeze_record_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"stage-2 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    consensus = pd.read_csv(consensus_path)
    blind_key = pd.read_csv(blind_key_path)
    reviewed_spans = pd.read_csv(reviewed_spans_path)
    line_index = pd.read_csv(line_index_path)
    evidence_a, omissions_a = build_stage2_evidence_forms(
        consensus,
        blind_key,
        reviewed_spans,
        line_index,
        reviewer_id="A",
    )
    evidence_b, omissions_b = build_stage2_evidence_forms(
        consensus,
        blind_key,
        reviewed_spans,
        line_index,
        reviewer_id="B",
    )
    evidence_a = _shuffle(evidence_a, seed=REVIEWER_A_ORDER_SEED)
    evidence_b = _shuffle(evidence_b, seed=REVIEWER_B_ORDER_SEED)
    omissions_a = _shuffle(omissions_a, seed=REVIEWER_A_ORDER_SEED + 1)
    omissions_b = _shuffle(omissions_b, seed=REVIEWER_B_ORDER_SEED + 1)

    _write(evidence_a, output_dir / "reviewer_A_stage2_evidence_blank.csv")
    _write(evidence_b, output_dir / "reviewer_B_stage2_evidence_blank.csv")
    _write(omissions_a, output_dir / "reviewer_A_stage2_omission_blank.csv")
    _write(omissions_b, output_dir / "reviewer_B_stage2_omission_blank.csv")
    (output_dir / "README.md").write_text(
        """# D-P证据审核第二阶段

本目录只有在第一阶段人工共识SHA-256冻结后才能生成。`stage2_evidence`表中的每一行
对应reviewed C5 spans源文件中的一个实际证据片段；行数不由D-P count或3+推算。
`stage2_omission`表中的每一行对应第一阶段人工共识记录的一项证据事件，用于判断
C5是否覆盖以及是否构成重要遗漏。主样本、富集样本和哨兵的统计归属仍由owner key
控制，评价者表中不显示sample_type。
""",
        encoding="utf-8",
    )
    files = [path for path in sorted(output_dir.iterdir()) if path.is_file()]
    manifest: dict[str, object] = {
        "status": "stage2_evidence_review_ready_not_scored",
        "consensus_freeze_status": freeze_record["status"],
        "actual_c5_span_row_n": int(len(evidence_a)),
        "human_reference_evidence_row_n": int(len(omissions_a)),
        "count_values_used_to_infer_span_rows": False,
        "reviewer_order_seeds": {
            "A": REVIEWER_A_ORDER_SEED,
            "B": REVIEWER_B_ORDER_SEED,
        },
        "inputs": {
            "consensus_sha256": _sha256(consensus_path),
            "freeze_record_sha256": _sha256(freeze_record_path),
            "blind_key_sha256": _sha256(blind_key_path),
            "reviewed_spans_sha256": _sha256(reviewed_spans_path),
            "line_index_sha256": _sha256(line_index_path),
        },
        "output_hashes": {path.name: _sha256(path) for path in files},
    }
    (output_dir / "stage2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare packet-3 stage-2 evidence review")
    parser.add_argument(
        "--consensus",
        type=Path,
        default=DEFAULT_PRIVATE_PACKET / "stage1" / "consensus_labels.csv",
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=DEFAULT_PRIVATE_PACKET / "stage1" / "consensus_freeze.json",
    )
    parser.add_argument(
        "--blind-key",
        type=Path,
        default=DEFAULT_PRIVATE_PACKET / "blind_id_key.csv",
    )
    parser.add_argument("--reviewed-spans", type=Path, default=DEFAULT_REVIEWED_SPANS_PATH)
    parser.add_argument(
        "--line-index",
        type=Path,
        default=DEFAULT_PUBLIC_PACKET / "blind_case_lines.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_PRIVATE_PACKET / "stage2"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare_stage2(
        consensus_path=args.consensus,
        freeze_record_path=args.freeze_record,
        blind_key_path=args.blind_key,
        reviewed_spans_path=args.reviewed_spans,
        line_index_path=args.line_index,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "actual_c5_span_row_n": manifest["actual_c5_span_row_n"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
