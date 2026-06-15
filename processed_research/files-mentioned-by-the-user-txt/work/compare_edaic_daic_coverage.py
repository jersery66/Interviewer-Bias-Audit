from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(r"F:\数据库\E-daic")
EXPECTED = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


def ids_from_archive_dir(path: Path) -> set[int]:
    out: set[int] = set()
    for p in path.glob("*_P.tar.gz"):
        m = re.match(r"(\d+)_P\.tar\.gz$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def ids_from_extracted_dir(path: Path) -> set[int]:
    out: set[int] = set()
    for p in path.glob("*_P.tar"):
        m = re.match(r"(\d+)_P\.tar$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def read_ids_csv(path: Path) -> tuple[list[str], list[int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        ids: list[int] = []
        for row in rdr:
            val = (
                row.get("Participant_ID")
                or row.get("participant_id")
                or row.get("Participant")
                or row.get("Participant_ID ")
                or next(iter(row.values()))
            )
            try:
                ids.append(int(float(str(val).strip())))
            except Exception:
                pass
        return rdr.fieldnames or [], ids


def transcript_info(pid: int) -> dict[str, object]:
    fp = ROOT / "data" / "data" / f"{pid}_P.tar" / f"{pid}_P" / f"{pid}_Transcript.csv"
    if not fp.exists():
        return {"pid": pid, "exists": False}
    with fp.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        speakers = set()
        n = 0
        sample = None
        for row in rdr:
            n += 1
            sp = row.get("speaker") or row.get("Speaker") or row.get("person") or ""
            speakers.add(sp)
            if sample is None:
                sample = row
    return {
        "pid": pid,
        "exists": True,
        "columns": rdr.fieldnames,
        "nrows": n,
        "speakers": sorted(speakers),
        "sample": sample,
    }


def main() -> None:
    expected_set = set(EXPECTED)
    archives = ids_from_archive_dir(ROOT / "E-daic" / "data")
    extracted = ids_from_extracted_dir(ROOT / "data" / "data")
    print("DAIC expected sessions:", len(EXPECTED), EXPECTED[:3], EXPECTED[-3:])
    print("E-DAIC archives total:", len(archives), "minmax:", (min(archives), max(archives)))
    print("E-DAIC extracted total:", len(extracted), "minmax:", (min(extracted), max(extracted)))
    print("Missing DAIC IDs in E-DAIC archives:", sorted(expected_set - archives))
    print("Missing DAIC IDs in extracted dirs:", sorted(expected_set - extracted))
    print("Extra E-DAIC IDs >=600:", len([x for x in archives if x >= 600]))

    split_union: set[int] = set()
    for split_name in ["train", "dev", "test"]:
        cols, ids = read_ids_csv(ROOT / "labels" / f"{split_name}_split.csv")
        ids_set = set(ids)
        split_union |= ids_set
        print(
            f"{split_name}_split:",
            "cols=", cols,
            "n=", len(ids_set),
            "DAIC_range_n=", len(ids_set & expected_set),
            "AI_600plus_n=", len([x for x in ids_set if x >= 600]),
        )
    print("DAIC IDs missing from E-DAIC splits:", sorted(expected_set - split_union))

    cols, label_ids = read_ids_csv(ROOT / "labels" / "Detailed_PHQ8_Labels.csv")
    label_set = set(label_ids)
    print(
        "Detailed_PHQ8_Labels:",
        "cols=", cols,
        "rows=", len(label_ids),
        "unique_ids=", len(label_set),
        "DAIC_range_labels=", len(label_set & expected_set),
    )
    print("DAIC IDs missing PHQ labels:", sorted(expected_set - label_set))

    for pid in [300, 373, 451, 458, 480, 492, 600]:
        print("TRANSCRIPT_INFO", transcript_info(pid))


if __name__ == "__main__":
    main()
