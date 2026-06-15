from __future__ import annotations

import argparse
import csv
import html.parser
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


BASES = {
    "edaic_root": "https://dcapswoz.ict.usc.edu/wwwedaic/",
    "edaic_data": "https://dcapswoz.ict.usc.edu/wwwedaic/data/",
    "edaic_labels": "https://dcapswoz.ict.usc.edu/wwwedaic/labels/",
    "daicwoz_root": "https://dcapswoz.ict.usc.edu/wwwdaicwoz/",
}


@dataclass
class RemoteItem:
    dataset: str
    section: str
    filename: str
    url: str
    size_text: str
    expected_bytes: Optional[int]


@dataclass
class AuditItem:
    dataset: str
    section: str
    filename: str
    url: str
    size_text: str
    expected_bytes: Optional[int]
    status: str
    local_path: str
    local_bytes: Optional[int]
    note: str


class IndexParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: Optional[str] = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "a":
            attrs_d = dict(attrs)
            self._href = attrs_d.get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = "".join(self._text).strip()
            self.links.append((self._href, text))
            self._href = None
            self._text = []


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_size_to_bytes(size_text: str) -> Optional[int]:
    text = size_text.strip()
    if not text or text == "-":
        return None
    m = re.fullmatch(r"([0-9.]+)\s*([KMGTP]?)", text, flags=re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[unit]
    return int(value * mult)


def parse_index(url: str, dataset: str, section: str) -> list[RemoteItem]:
    html = fetch(url)
    parser = IndexParser()
    parser.feed(html)
    rows: list[RemoteItem] = []
    for href, text in parser.links:
        if href in ("../", "/") or text in ("Parent Directory", "Name", "Last modified", "Size", "Description"):
            continue
        if href.endswith("/"):
            continue
        # Apache autoindex line keeps date and size after the link. Use the whole
        # line containing the filename, then capture the last size-like token.
        line_match = re.search(re.escape(text) + r".*?(?P<size>[0-9.]+\s*[KMGTP]?|-)\s*(?:\n|$)", html, flags=re.I)
        size_text = line_match.group("size").replace(" ", "") if line_match else ""
        rows.append(
            RemoteItem(
                dataset=dataset,
                section=section,
                filename=text,
                url=url + href,
                size_text=size_text,
                expected_bytes=parse_size_to_bytes(size_text),
            )
        )
    return rows


def scan_local(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    file_mapping: dict[str, list[Path]] = {}
    dir_mapping: dict[str, list[Path]] = {}
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        for dirname in dirnames:
            dir_mapping.setdefault(dirname.lower(), []).append(current / dirname)
        for filename in filenames:
            file_mapping.setdefault(filename.lower(), []).append(current / filename)
    return file_mapping, dir_mapping


def find_extracted_participant(
    root: Path,
    dir_mapping: dict[str, list[Path]],
    dataset: str,
    filename: str,
) -> Optional[Path]:
    stem = filename
    for suffix in (".tar.gz", ".zip", ".tgz", ".tar"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not re.fullmatch(r"\d+_P", stem):
        return None
    if dataset == "eDAIC":
        candidates = [
            root / "E-daic" / "data" / "data" / f"{stem}.tar" / stem,
            root / "E-daic" / "data" / f"{stem}.tar" / stem,
            root / "E-daic" / "data" / stem,
        ]
    elif dataset == "DAIC-WOZ":
        candidates = [
            root / "DAIC-WOZ" / stem,
            root / "DAIC-WOZ" / f"{stem}.zip" / stem,
            root / "DAICWOZ" / stem,
            root / "DAICWOZ" / f"{stem}.zip" / stem,
            root / "DAIC-WOZ原始数据" / stem,
            root / "DAIC-WOZ原始数据" / f"{stem}.zip" / stem,
            root / "压缩包" / "DAIC-WOZ" / stem,
            root / "压缩包" / "DAICWOZ" / stem,
        ]
    else:
        candidates = []
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    # Fallback index check, scoped by dataset-looking ancestor paths only.
    for candidate in dir_mapping.get(stem.lower(), []):
        parts = {p.lower() for p in candidate.parts}
        joined = str(candidate).lower()
        if dataset == "eDAIC" and ("e-daic" in parts or "edaic" in joined):
            return candidate
        if dataset == "DAIC-WOZ" and ("daic-woz" in joined or "daicwoz" in joined):
            return candidate
    return None


def choose_local_file(paths: Iterable[Path], expected_bytes: Optional[int]) -> tuple[Optional[Path], Optional[int], str]:
    paths = list(paths)
    if not paths:
        return None, None, "not found"
    if expected_bytes is None:
        path = max(paths, key=lambda p: p.stat().st_size)
        return path, path.stat().st_size, "found, size not checked"
    # Apache index sizes are rounded, so allow 2 MiB or 1%, whichever is larger.
    tolerance = max(2 * 1024 * 1024, int(expected_bytes * 0.01))
    best = min(paths, key=lambda p: abs(p.stat().st_size - expected_bytes))
    local_bytes = best.stat().st_size
    if abs(local_bytes - expected_bytes) <= tolerance:
        return best, local_bytes, "size within rounded-index tolerance"
    return best, local_bytes, "size mismatch against rounded index"


def audit(root: Path, remote_items: list[RemoteItem]) -> list[AuditItem]:
    files, dirs = scan_local(root)
    rows: list[AuditItem] = []
    for item in remote_items:
        local_paths = files.get(item.filename.lower(), [])
        local_path, local_bytes, note = choose_local_file(local_paths, item.expected_bytes)
        if local_path is not None:
            status = "present_file" if "mismatch" not in note else "size_mismatch"
            rows.append(
                AuditItem(
                    **asdict(item),
                    status=status,
                    local_path=str(local_path),
                    local_bytes=local_bytes,
                    note=note,
                )
            )
            continue
        extracted = find_extracted_participant(root, dirs, item.dataset, item.filename)
        if extracted is not None:
            rows.append(
                AuditItem(
                    **asdict(item),
                    status="present_extracted",
                    local_path=str(extracted),
                    local_bytes=None,
                    note="archive absent but extracted participant directory exists",
                )
            )
        else:
            rows.append(
                AuditItem(
                    **asdict(item),
                    status="missing",
                    local_path="",
                    local_bytes=None,
                    note="not found as archive or extracted participant directory",
                )
            )
    return rows


def write_csv(path: Path, rows: list[AuditItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    remote_items: list[RemoteItem] = []
    remote_items += parse_index(BASES["edaic_root"], "eDAIC", "root")
    remote_items += parse_index(BASES["edaic_data"], "eDAIC", "data")
    remote_items += parse_index(BASES["edaic_labels"], "eDAIC", "labels")
    remote_items += parse_index(BASES["daicwoz_root"], "DAIC-WOZ", "root")

    rows = audit(args.root, remote_items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dcapswoz_audit.csv", rows)

    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        summary.setdefault(row.dataset, {}).setdefault(row.status, 0)
        summary[row.dataset][row.status] += 1
    summary_path = args.output_dir / "dcapswoz_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_path = args.output_dir / "dcapswoz_missing_urls.txt"
    with missing_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.status in {"missing", "size_mismatch"}:
                f.write(row.url + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"audit_csv={args.output_dir / 'dcapswoz_audit.csv'}")
    print(f"missing_urls={missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
