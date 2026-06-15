from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
import queue
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


CHUNK_SIZE = 8 * 1024 * 1024
LOG_LOCK = threading.Lock()


@dataclass
class DownloadItem:
    url: str
    filename: str
    dataset: str
    expected_bytes: Optional[int]
    dest: Path


def infer_item(row: dict[str, str], dest_root: Path) -> DownloadItem:
    url = row["url"]
    filename = url.rsplit("/", 1)[-1].replace("%20", " ")
    dataset = "eDAIC" if "wwwedaic" in url else "DAIC-WOZ"
    if dataset == "eDAIC":
        dest_dir = dest_root / "E-daic"
    else:
        dest_dir = dest_root / "DAIC-WOZ"
    content_length = row.get("content_length") or row.get("expected_bytes") or ""
    expected = int(content_length) if str(content_length).strip().isdigit() else None
    return DownloadItem(
        url=url,
        filename=filename,
        dataset=dataset,
        expected_bytes=expected,
        dest=dest_dir / filename,
    )


def load_items(head_csv: Path, dest_root: Path) -> list[DownloadItem]:
    with head_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [infer_item(row, dest_root) for row in rows]


def log_event(log_path: Path, event: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def file_complete(path: Path, expected_bytes: Optional[int]) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if expected_bytes is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_bytes


def open_with_resume(url: str, start: int):
    headers = {"User-Agent": "Mozilla/5.0"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=90)


def download_one(item: DownloadItem, log_path: Path, deadline: float, retries: int) -> str:
    item.dest.parent.mkdir(parents=True, exist_ok=True)
    if file_complete(item.dest, item.expected_bytes):
        return "already_complete"

    part = item.dest.with_suffix(item.dest.suffix + ".part")
    if item.dest.exists() and not file_complete(item.dest, item.expected_bytes):
        # curl-style interrupted downloads leave the partial content at the
        # final filename. Convert it back to the Python downloader's .part form.
        if not part.exists() or item.dest.stat().st_size > part.stat().st_size:
            os.replace(item.dest, part)
        else:
            item.dest.unlink()
    for attempt in range(1, retries + 1):
        if time.monotonic() >= deadline:
            return "time_limit"
        start = part.stat().st_size if part.exists() else 0
        mode = "ab" if start > 0 else "wb"
        try:
            with open_with_resume(item.url, start) as resp:
                status = getattr(resp, "status", None)
                if start > 0 and status != 206:
                    start = 0
                    mode = "wb"
                log_event(
                    log_path,
                    {
                        "event": "start_file",
                        "url": item.url,
                        "dest": str(item.dest),
                        "expected_bytes": item.expected_bytes,
                        "resume_from": start,
                        "http_status": status,
                        "attempt": attempt,
                    },
                )
                with part.open(mode + "") as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        if time.monotonic() >= deadline:
                            log_event(
                                log_path,
                                {
                                    "event": "checkpoint",
                                    "url": item.url,
                                    "dest": str(item.dest),
                                    "part_bytes": part.stat().st_size,
                                    "expected_bytes": item.expected_bytes,
                                },
                            )
                            return "time_limit"
            part_bytes = part.stat().st_size if part.exists() else 0
            if item.expected_bytes is not None and part_bytes != item.expected_bytes:
                log_event(
                    log_path,
                    {
                        "event": "size_incomplete",
                        "url": item.url,
                        "dest": str(item.dest),
                        "part_bytes": part_bytes,
                        "expected_bytes": item.expected_bytes,
                    },
                )
                return "incomplete"
            os.replace(part, item.dest)
            log_event(
                log_path,
                {
                    "event": "complete_file",
                    "url": item.url,
                    "dest": str(item.dest),
                    "bytes": item.dest.stat().st_size,
                    "expected_bytes": item.expected_bytes,
                },
            )
            return "downloaded"
        except Exception as exc:
            log_event(
                log_path,
                {
                    "event": "error",
                    "url": item.url,
                    "dest": str(item.dest),
                    "attempt": attempt,
                    "error": repr(exc),
                },
            )
            time.sleep(min(10, attempt * 2))
    return "failed"


def write_status(items: list[DownloadItem], status_path: Path) -> dict[str, object]:
    rows = []
    complete = 0
    part_bytes = 0
    total_known = 0
    for item in items:
        part = item.dest.with_suffix(item.dest.suffix + ".part")
        dest_bytes = item.dest.stat().st_size if item.dest.exists() else 0
        p_bytes = part.stat().st_size if part.exists() else 0
        is_complete = file_complete(item.dest, item.expected_bytes)
        complete += int(is_complete)
        if item.expected_bytes:
            total_known += item.expected_bytes
        part_bytes += dest_bytes + p_bytes
        rows.append(
            {
                "dataset": item.dataset,
                "filename": item.filename,
                "url": item.url,
                "dest": str(item.dest),
                "expected_bytes": item.expected_bytes,
                "dest_bytes": dest_bytes,
                "part_bytes": p_bytes,
                "complete": is_complete,
            }
        )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = status_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(items),
        "complete_count": complete,
        "remaining_count": len(items) - complete,
        "known_total_gb": round(total_known / (1024**3), 2),
        "downloaded_or_part_gb": round(part_bytes / (1024**3), 2),
        "status_csv": str(csv_path),
    }
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def verify_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            return "ok" if bad is None else f"bad_member:{bad}"
    except Exception as exc:
        return f"error:{exc!r}"


def verify(items: list[DownloadItem], output_dir: Path) -> int:
    rows = []
    for item in items:
        complete = file_complete(item.dest, item.expected_bytes)
        archive_check = ""
        if complete and item.dest.suffix.lower() == ".zip":
            archive_check = verify_zip(item.dest)
        elif complete and item.dest.suffix.lower() in {".gz", ".pdf", ".csv", ".txt"}:
            archive_check = "size_ok"
        rows.append(
            {
                "dataset": item.dataset,
                "filename": item.filename,
                "dest": str(item.dest),
                "expected_bytes": item.expected_bytes,
                "actual_bytes": item.dest.stat().st_size if item.dest.exists() else None,
                "size_complete": complete,
                "archive_check": archive_check,
            }
        )
    path = output_dir / "download_integrity_report.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    failures = [r for r in rows if not r["size_complete"] or (r["archive_check"] and r["archive_check"] not in {"ok", "size_ok"})]
    print(json.dumps({"integrity_report": str(path), "failure_count": len(failures)}, ensure_ascii=False, indent=2))
    return len(failures)


def run_parallel(
    items: list[DownloadItem],
    log_path: Path,
    deadline: float,
    retries: int,
    workers: int,
) -> dict[str, int]:
    actions = {"already_complete": 0, "downloaded": 0, "time_limit": 0, "incomplete": 0, "failed": 0}
    actions_lock = threading.Lock()
    q: queue.Queue[DownloadItem] = queue.Queue()
    for item in items:
        if not file_complete(item.dest, item.expected_bytes):
            q.put(item)

    def worker() -> None:
        while time.monotonic() < deadline:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            try:
                result = download_one(item, log_path, deadline, retries)
                with actions_lock:
                    actions[result] = actions.get(result, 0) + 1
                if result == "time_limit":
                    return
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        remaining = max(1, deadline - time.monotonic() + 30)
        thread.join(timeout=remaining)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-csv", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    items = load_items(args.head_csv, args.dest_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "download_status.json"
    if args.verify:
        return verify(items, args.output_dir)

    deadline = time.monotonic() + args.max_seconds
    log_path = args.output_dir / "download_log.jsonl"
    actions = run_parallel(items, log_path, deadline, args.retries, args.workers)
    summary = write_status(items, status_path)
    summary["actions"] = actions
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["remaining_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
