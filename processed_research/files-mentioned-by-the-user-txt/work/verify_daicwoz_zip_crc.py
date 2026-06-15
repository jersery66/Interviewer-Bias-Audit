from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOTS = [Path(r"F:\\"), Path(r"F:\\数据库\\DAIC-WOZ")]
EXPECTED = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


def find_zip(pid: int) -> Path | None:
    pattern = re.compile(rf"(^|.*?){pid}_P\.zip$", re.I)
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_file() and pattern.fullmatch(p.name):
                return p
    return None


def main() -> int:
    rows = []
    for pid in EXPECTED:
        path = find_zip(pid)
        if path is None:
            rows.append({"pid": pid, "status": "missing", "path": ""})
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                names = zf.namelist()
            rows.append(
                {
                    "pid": pid,
                    "status": "ok" if bad is None else "bad_member",
                    "bad_member": bad or "",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "members": len(names),
                }
            )
        except Exception as exc:
            rows.append({"pid": pid, "status": "error", "path": str(path), "error": repr(exc)})
    out = Path("outputs") / "daicwoz_f_audit" / "crc_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "total": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "failures": [r for r in rows if r["status"] != "ok"],
        "report": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
