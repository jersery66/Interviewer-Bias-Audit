from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.source_definition_audit import audit_domain_sources, write_domain_source_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit D-P provenance and gate a full-transcript D-All input"
    )
    parser.add_argument("--participant-text", required=True, type=Path)
    parser.add_argument("--interviewer-text", required=True, type=Path)
    parser.add_argument("--full-text", required=True, type=Path)
    parser.add_argument("--reviewed-spans", required=True, type=Path)
    parser.add_argument("--domain-count", required=True, type=Path)
    parser.add_argument("--d-all", type=Path, default=None)
    parser.add_argument("--d-all-provenance", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = audit_domain_sources(
        participant_text_path=args.participant_text,
        interviewer_text_path=args.interviewer_text,
        full_text_path=args.full_text,
        reviewed_spans_path=args.reviewed_spans,
        domain_count_path=args.domain_count,
        d_all_path=args.d_all,
        d_all_provenance_path=args.d_all_provenance,
    )
    write_domain_source_audit(result, args.output)
    print(json.dumps({"d_p": result["d_p"]["status"], "d_all": result["d_all"]["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
