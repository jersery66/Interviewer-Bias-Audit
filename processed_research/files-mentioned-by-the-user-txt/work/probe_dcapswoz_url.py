from __future__ import annotations

import argparse
import json
import urllib.request


def probe(url: str, method: str, range_header: str | None = None, max_bytes: int = 1024 * 1024) -> dict[str, object]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = b""
            if method != "HEAD":
                data = resp.read(max_bytes)
            return {
                "method": method,
                "range": range_header,
                "status": resp.status,
                "content_length": resp.headers.get("Content-Length"),
                "content_range": resp.headers.get("Content-Range"),
                "bytes_read": len(data),
                "first_bytes": data[:8].hex(),
                "error": "",
            }
    except Exception as exc:
        return {
            "method": method,
            "range": range_header,
            "status": None,
            "content_length": None,
            "content_range": None,
            "bytes_read": 0,
            "first_bytes": "",
            "error": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    rows = [
        probe(args.url, "HEAD"),
        probe(args.url, "GET", None),
        probe(args.url, "GET", "bytes=100663296-101711871"),
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
