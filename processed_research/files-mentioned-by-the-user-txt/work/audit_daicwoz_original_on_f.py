from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(r"F:\\")
EXPECTED = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]
EXCLUDE_PARTS = {"E-daic", "E-DAIC", "edaic"}


def is_excluded(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(p.lower() in parts for p in EXCLUDE_PARTS)


def transcript_has_speaker(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            header = f.readline()
            delimiter = "\t" if "\t" in header else ","
            cols = [c.strip().lower() for c in header.strip().split(delimiter)]
        if {"speaker", "value"}.issubset(set(cols)):
            return True, "speaker/value transcript"
        return False, "transcript without speaker/value columns: " + ",".join(cols)
    except Exception as exc:
        return False, f"transcript read error: {exc!r}"


def zip_has_speaker(path: Path, pid: int) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            candidates = [
                n for n in names
                if re.search(rf"(^|/){pid}_TRANSCRIPT\.csv$", n, flags=re.I)
            ]
            if not candidates:
                return False, "zip valid but transcript not found"
            with zf.open(candidates[0]) as f:
                header = f.readline().decode("utf-8-sig", errors="replace")
            delimiter = "\t" if "\t" in header else ","
            cols = [c.strip().lower() for c in header.strip().split(delimiter)]
            if {"speaker", "value"}.issubset(set(cols)):
                return True, "zip valid with speaker/value transcript"
            return False, "zip transcript without speaker/value columns: " + ",".join(cols)
    except Exception as exc:
        return False, f"zip error: {exc!r}"


def candidate_bases() -> list[Path]:
    bases = [
        ROOT,
        ROOT / "数据库" / "DAIC-WOZ",
        ROOT / "DAIC-WOZ",
        ROOT / "DAICWOZ",
    ]
    out = []
    for base in bases:
        if base.exists() and base.is_dir() and base not in out:
            out.append(base)
    return out


def scan_base(base: Path, hits: dict[int, list[dict[str, object]]], partials: list[dict[str, object]]) -> None:
    try:
        entries = list(base.iterdir())
    except Exception:
        return
    for entry in entries:
        if is_excluded(entry):
            continue
        if entry.is_dir():
            m = re.fullmatch(r"(\d+)_P", entry.name, flags=re.I)
            if not m:
                continue
            pid = int(m.group(1))
            if pid not in hits:
                continue
            transcript = entry / f"{pid}_TRANSCRIPT.csv"
            if transcript.exists():
                ok, note = transcript_has_speaker(transcript)
            else:
                ok, note = False, "folder without *_TRANSCRIPT.csv"
            hits[pid].append({"kind": "folder", "path": str(entry), "ok": ok, "note": note})
        elif entry.is_file():
            m = re.search(r"(\d+)_P\.zip(\.crdownload|\.part)?$", entry.name, flags=re.I)
            if not m:
                continue
            pid = int(m.group(1))
            if pid not in hits:
                continue
            suffix = m.group(2)
            if suffix:
                partials.append({"pid": pid, "path": str(entry), "bytes": entry.stat().st_size})
                hits[pid].append({"kind": "partial", "path": str(entry), "ok": False, "note": "partial download"})
            else:
                ok, note = zip_has_speaker(entry, pid)
                hits[pid].append({"kind": "zip", "path": str(entry), "ok": ok, "note": note, "bytes": entry.stat().st_size})


def scan() -> tuple[dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    hits: dict[int, list[dict[str, object]]] = {pid: [] for pid in EXPECTED}
    partials: list[dict[str, object]] = []
    for base in candidate_bases():
        scan_base(base, hits, partials)
    return hits, partials


def main() -> None:
    hits, partials = scan()
    complete = []
    missing = []
    questionable = []
    for pid in EXPECTED:
        pid_hits = hits[pid]
        ok_hits = [h for h in pid_hits if h["ok"]]
        if ok_hits:
            complete.append(pid)
        elif pid_hits:
            questionable.append({"pid": pid, "hits": pid_hits})
            missing.append(pid)
        else:
            missing.append(pid)

    outdir = Path("outputs") / "daicwoz_f_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "expected_count": len(EXPECTED),
                "complete_count": len(complete),
                "missing_count": len(missing),
                "questionable_count": len(questionable),
                "complete_minmax": [min(complete), max(complete)] if complete else None,
                "missing_first_30": missing[:30],
                "partials": partials,
                "scanned_bases": [str(p) for p in candidate_bases()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (outdir / "complete_ids.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(map(str, complete)) + ("\n" if complete else ""))
    with (outdir / "missing_ids.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(map(str, missing)) + ("\n" if missing else ""))
    with (outdir / "missing_urls.txt").open("w", encoding="utf-8") as f:
        for pid in missing:
            f.write(f"https://dcapswoz.ict.usc.edu/wwwdaicwoz/{pid}_P.zip\n")
    with (outdir / "questionable.json").open("w", encoding="utf-8") as f:
        json.dump(questionable, f, ensure_ascii=False, indent=2)
    print((outdir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
