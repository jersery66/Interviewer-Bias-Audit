from __future__ import annotations

import argparse
import csv
from pathlib import Path


def filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].replace("%20", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-csv", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-max", type=int, default=6)
    parser.add_argument("--dataset", choices=["all", "eDAIC", "DAIC-WOZ"], default="all")
    parser.add_argument("--basename-output", action="store_true")
    args = parser.parse_args()

    with args.head_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    lines = [
        "location",
        "fail",
        "continue-at = -",
        "retry = 8",
        "retry-delay = 3",
        "connect-timeout = 30",
        "parallel",
        f"parallel-max = {args.parallel_max}",
    ]

    for row in rows:
        url = row["url"]
        filename = filename_from_url(url)
        dataset = "eDAIC" if "wwwedaic" in url else "DAIC-WOZ"
        if args.dataset != "all" and dataset != args.dataset:
            continue
        if args.basename_output:
            dest = Path(filename)
        elif dataset == "eDAIC":
            dest = args.dest_root / "E-daic" / filename
        else:
            dest = args.dest_root / "DAIC-WOZ" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f'url = "{url}"')
        lines.append(f'output = "{dest}"')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
