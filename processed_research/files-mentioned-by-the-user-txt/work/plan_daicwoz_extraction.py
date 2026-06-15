from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOTS = [Path(r"F:\\"), Path(r"F:\\数据库\\DAIC-WOZ")]
TARGET = Path(r"F:\\数据库\\DAIC-WOZ")
EXPECTED = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


def find_zip(pid: int) -> Path | None:
    candidates = []
    pat = re.compile(rf".*{pid}_P\.zip$", re.I)
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_file() and pat.fullmatch(p.name):
                candidates.append(p)
    if not candidates:
        return None
    # Prefer normal filename over prefixed legacy download names.
    candidates.sort(key=lambda p: (p.name.lower() != f"{pid}_p.zip", len(p.name), str(p)))
    return candidates[0]


def folder_ok(pid: int) -> bool:
    t = TARGET / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
    if not t.exists():
        return False
    try:
        header = t.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    except Exception:
        return False
    delim = "\t" if "\t" in header else ","
    cols = {c.strip().lower() for c in header.split(delim)}
    return {"speaker", "value"}.issubset(cols)


def main() -> None:
    rows = []
    for pid in EXPECTED:
        z = find_zip(pid)
        if z is None:
            rows.append({"pid": pid, "status": "zip_missing"})
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                infos = zf.infolist()
                uncomp = sum(i.file_size for i in infos)
                members = len(infos)
        except Exception as exc:
            rows.append({"pid": pid, "status": "zip_error", "zip": str(z), "error": repr(exc)})
            continue
        rows.append(
            {
                "pid": pid,
                "status": "already_extracted" if folder_ok(pid) else "needs_extract",
                "zip": str(z),
                "compressed_bytes": z.stat().st_size,
                "uncompressed_bytes": uncomp,
                "members": members,
            }
        )
    needs = [r for r in rows if r["status"] == "needs_extract"]
    ok = [r for r in rows if r["status"] == "already_extracted"]
    missing = [r for r in rows if r["status"] != "needs_extract" and r["status"] != "already_extracted"]
    summary = {
        "expected": len(EXPECTED),
        "already_extracted": len(ok),
        "needs_extract": len(needs),
        "problem_count": len(missing),
        "compressed_total_gb": round(sum(r.get("compressed_bytes", 0) for r in rows) / (1024**3), 2),
        "uncompressed_total_gb": round(sum(r.get("uncompressed_bytes", 0) for r in rows) / (1024**3), 2),
        "uncompressed_needed_gb": round(sum(r.get("uncompressed_bytes", 0) for r in needs) / (1024**3), 2),
        "first_needs": [r["pid"] for r in needs[:20]],
        "problems": missing[:20],
    }
    outdir = Path("outputs") / "daicwoz_extract_plan"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "extract_plan.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
