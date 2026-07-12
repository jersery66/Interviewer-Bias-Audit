from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TEXT_SUFFIXES = {".csv", ".json", ".md", ".svg", ".tsv", ".txt", ".yaml", ".yml"}


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def compare_recomputations(
    first_root: Path,
    second_root: Path,
    first_run_id: str,
    second_run_id: str,
) -> dict[str, object]:
    first_files = {path.name: path for path in first_root.glob("*.csv")}
    second_files = {path.name: path for path in second_root.glob("*.csv")}
    names = sorted(set(first_files) | set(second_files))
    rows: list[dict[str, object]] = []
    for name in names:
        first = first_files.get(name)
        second = second_files.get(name)
        first_hash = normalized_sha256(first) if first else None
        second_hash = normalized_sha256(second) if second else None
        rows.append(
            {
                "file": name,
                "run1_sha256": first_hash,
                "run2_sha256": second_hash,
                "equal": first_hash is not None and first_hash == second_hash,
            }
        )
    all_equal = bool(rows) and all(bool(row["equal"]) for row in rows)
    return {
        "status": "PASS" if all_equal else "FAIL",
        "run1": first_run_id,
        "run2": second_run_id,
        "file_count": len(rows),
        "hash_normalization": "CRLF-to-LF for text artifacts; raw bytes for binary artifacts",
        "files": rows,
    }


def update_output_manifest(audit_dir: Path, artifact: Path) -> None:
    inventory_path = audit_dir / "output_manifest_sha256.csv"
    del artifact
    files = sorted(
        path
        for path in audit_dir.rglob("*")
        if path.is_file() and path.name != inventory_path.name
    )
    inventory = pd.DataFrame(
        [
            {
                "file": path.relative_to(audit_dir).as_posix(),
                "sha256": normalized_sha256(path),
            }
            for path in files
        ]
    )
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record deterministic core-output recomputation evidence.")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--first-run-id", required=True)
    parser.add_argument("--second-run-id", required=True)
    args = parser.parse_args()
    report = compare_recomputations(
        args.first,
        args.second,
        args.first_run_id,
        args.second_run_id,
    )
    artifact = args.audit_dir / "recomputation_verification.json"
    artifact.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    update_output_manifest(args.audit_dir, artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
