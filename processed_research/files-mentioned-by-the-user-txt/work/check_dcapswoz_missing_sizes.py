from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path


def head(url: str) -> dict[str, object]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            return {
                "url": url,
                "status": resp.status,
                "content_length": int(length) if length else None,
                "content_type": resp.headers.get("Content-Type"),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "status": exc.code,
            "content_length": None,
            "content_type": exc.headers.get("Content-Type") if exc.headers else None,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "url": url,
            "status": None,
            "content_length": None,
            "content_type": None,
            "error": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-urls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    urls = [line.strip() for line in args.missing_urls.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [head(url) for url in urls]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "missing_url_head.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "status", "content_length", "content_type", "error"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum((row["content_length"] or 0) for row in rows)
    summary = {
        "count": len(rows),
        "ok_count": sum(1 for row in rows if row["status"] == 200),
        "total_bytes_known": total,
        "total_gb_known": round(total / (1024**3), 2),
        "unknown_length_count": sum(1 for row in rows if row["content_length"] is None),
        "status_counts": {},
    }
    for row in rows:
        key = str(row["status"])
        summary["status_counts"][key] = summary["status_counts"].get(key, 0) + 1
    (args.output_dir / "missing_url_head_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
