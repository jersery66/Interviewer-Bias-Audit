from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError


CHUNK = 8 * 1024 * 1024
BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"


def expected_ids() -> list[int]:
    return [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


def transcript_ok_in_zip(path: Path, pid: int) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            candidates = [n for n in names if re.search(rf"(^|/){pid}_TRANSCRIPT\.csv$", n, re.I)]
            if not candidates:
                return False, "transcript missing"
            with zf.open(candidates[0]) as f:
                header = f.readline().decode("utf-8-sig", errors="replace")
            delim = "\t" if "\t" in header else ","
            cols = {c.strip().lower() for c in header.strip().split(delim)}
            if {"speaker", "value"}.issubset(cols):
                return True, "speaker transcript present"
            return False, "transcript lacks speaker/value"
    except Exception as exc:
        return False, f"zip error: {exc!r}"


def transcript_ok_in_folder(path: Path, pid: int) -> tuple[bool, str]:
    transcript = path / f"{pid}_TRANSCRIPT.csv"
    if not transcript.exists():
        return False, "folder transcript missing"
    try:
        header = transcript.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
        delim = "\t" if "\t" in header else ","
        cols = {c.strip().lower() for c in header.strip().split(delim)}
        if {"speaker", "value"}.issubset(cols):
            return True, "speaker transcript present"
        return False, "transcript lacks speaker/value"
    except Exception as exc:
        return False, f"folder transcript error: {exc!r}"


def is_complete(dest_dir: Path, pid: int) -> tuple[bool, str]:
    folder = dest_dir / f"{pid}_P"
    if folder.exists() and folder.is_dir():
        ok, note = transcript_ok_in_folder(folder, pid)
        if ok:
            return True, f"folder ok: {folder}"
    zip_path = dest_dir / f"{pid}_P.zip"
    if zip_path.exists():
        ok, note = transcript_ok_in_zip(zip_path, pid)
        if ok:
            return True, f"zip ok: {zip_path}"
        return False, f"zip present but invalid: {note}"
    return False, "missing"


def remote_size(pid: int, retries: int = 5, delay: int = 60) -> int:
    url = f"{BASE_URL}/{pid}_P.zip"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"HEAD failed for {pid}_P.zip after {retries} attempts: {last_exc!r}")


def download_pid(dest_dir: Path, pid: int, log_path: Path, retries: int = 5, delay: int = 60) -> dict[str, object]:
    dest = dest_dir / f"{pid}_P.zip"
    part = dest_dir / f"{pid}_P.zip.part"
    size = remote_size(pid, retries=retries, delay=delay)
    if dest.exists() and dest.stat().st_size == size:
        ok, note = transcript_ok_in_zip(dest, pid)
        return {"pid": pid, "status": "already_complete" if ok else "invalid_existing", "bytes": dest.stat().st_size, "note": note}
    if dest.exists() and dest.stat().st_size != size:
        if not part.exists() or dest.stat().st_size > part.stat().st_size:
            os.replace(dest, part)
        else:
            dest.unlink()
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    url = f"{BASE_URL}/{pid}_P.zip"
    req = urllib.request.Request(url, headers=headers)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": "start", "pid": pid, "url": url, "start": start, "expected": size}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                mode = "ab" if start and getattr(resp, "status", None) == 206 else "wb"
                with part.open(mode) as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                return {"pid": pid, "status": "failed", "bytes": part.stat().st_size if part.exists() else 0, "expected": size, "note": repr(last_exc)}
            time.sleep(delay)
            start = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if start:
                headers["Range"] = f"bytes={start}-"
            req = urllib.request.Request(url, headers=headers)
    actual = part.stat().st_size
    if actual != size:
        return {"pid": pid, "status": "incomplete", "bytes": actual, "expected": size, "note": "size mismatch after transfer"}
    os.replace(part, dest)
    ok, note = transcript_ok_in_zip(dest, pid)
    return {"pid": pid, "status": "downloaded" if ok else "invalid_download", "bytes": dest.stat().st_size, "expected": size, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-dir", type=Path, default=Path(r"F:\\"))
    ap.add_argument("--ids-file", type=Path, default=Path("outputs/daicwoz_f_audit/missing_ids.txt"))
    ap.add_argument("--max-files", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/daicwoz_f_download"))
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--retry-delay", type=int, default=60)
    ap.add_argument("--cooldown", type=int, default=60)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(x.strip()) for x in args.ids_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = []
    for pid in ids:
        ok, note = is_complete(args.dest_dir, pid)
        if ok:
            results.append({"pid": pid, "status": "skip_complete", "note": note})
            continue
        result = download_pid(args.dest_dir, pid, args.output_dir / "download_log.jsonl", retries=args.retries, delay=args.retry_delay)
        results.append(result)
        if result["status"] == "failed":
            break
        if result["status"] in {"downloaded", "already_complete", "invalid_download", "incomplete"}:
            # One server-facing transfer per loop; rerun for the next file.
            if len([r for r in results if r["status"] not in {"skip_complete"}]) >= args.max_files:
                break
            if args.cooldown > 0:
                time.sleep(args.cooldown)
    (args.output_dir / "last_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
