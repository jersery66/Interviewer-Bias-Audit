from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


F_ROOT = Path(r"F:\\")
EDAIC_ROOT = Path(r"F:\\数据库\\E-daic")
EDAIC_DATA = EDAIC_ROOT / "data" / "data"
DAIC_DIR = Path(r"F:\\数据库\\DAIC-WOZ")
OUTDIR = Path("outputs") / "daicwoz_edaic_dedupe"

EXPECTED_IDS = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


@dataclass(frozen=True)
class SourceFile:
    pid: int
    source_kind: str
    source_path: Path
    member_name: str
    basename: str
    size: int


def human_size(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num} B"


def iter_top_level_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    try:
        return (p for p in base.iterdir() if p.is_file())
    except OSError:
        return []


def discover_sources() -> dict[int, dict[str, object]]:
    sources: dict[int, dict[str, object]] = {pid: {"zips": [], "folders": []} for pid in EXPECTED_IDS}
    search_dirs = [F_ROOT, DAIC_DIR]
    for base in search_dirs:
        for path in iter_top_level_files(base):
            m = re.search(r"(\d+)_P\.zip$", path.name, flags=re.IGNORECASE)
            if not m:
                continue
            pid = int(m.group(1))
            if pid in sources:
                sources[pid]["zips"].append(path)

    for pid in EXPECTED_IDS:
        for folder in [F_ROOT / f"{pid}_P", DAIC_DIR / f"{pid}_P"]:
            if folder.exists() and folder.is_dir():
                sources[pid]["folders"].append(folder)

    return sources


def choose_source(pid: int, entry: dict[str, object]) -> tuple[str, Path] | None:
    zips: list[Path] = sorted(entry["zips"], key=lambda p: (len(str(p)), str(p).lower()))
    folders: list[Path] = sorted(entry["folders"], key=lambda p: (len(str(p)), str(p).lower()))
    if zips:
        return "zip", zips[0]
    if folders:
        return "folder", folders[0]
    return None


def source_files_for(pid: int, kind: str, path: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    if kind == "folder":
        for child in path.iterdir():
            if child.is_file():
                files.append(
                    SourceFile(
                        pid=pid,
                        source_kind=kind,
                        source_path=path,
                        member_name=child.name,
                        basename=child.name,
                        size=child.stat().st_size,
                    )
                )
        return files

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if not basename:
                continue
            files.append(
                SourceFile(
                    pid=pid,
                    source_kind=kind,
                    source_path=path,
                    member_name=info.filename,
                    basename=basename,
                    size=info.file_size,
                )
            )
    return files


def edaic_files_for(pid: int) -> dict[str, dict[str, object]]:
    base = EDAIC_DATA / f"{pid}_P.tar" / f"{pid}_P"
    out: dict[str, dict[str, object]] = {}
    if not base.exists():
        return out
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        key = path.name.lower()
        # Keep the first hit by basename; this is enough for detecting Windows name
        # collisions and direct duplicate files.
        out.setdefault(
            key,
            {
                "path": str(path),
                "size": path.stat().st_size,
                "relative": str(path.relative_to(base)),
            },
        )
    return out


def classify_file(sf: SourceFile, edaic_map: dict[str, dict[str, object]]) -> str:
    lower = sf.basename.lower()
    hit = edaic_map.get(lower)
    if hit is not None:
        if int(hit["size"]) == sf.size:
            return "duplicate_same_name_size"
        return "name_conflict_different_size"

    if re.fullmatch(rf"{sf.pid}_transcript\.csv", lower):
        return "needed_original_speaker_transcript"
    if re.fullmatch(rf"{sf.pid}_audio\.wav", lower):
        return "audio_not_in_edaic_by_name"
    return "daic_only_optional_feature"


def has_speaker_columns_from_zip(zip_path: Path, member_name: str) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            header = f.readline().decode("utf-8-sig", errors="replace")
    delimiter = "\t" if "\t" in header else ","
    cols = {c.strip().lower() for c in header.strip().split(delimiter)}
    return {"speaker", "value"}.issubset(cols)


def has_speaker_columns_from_folder(folder: Path, member_name: str) -> bool:
    with (folder / member_name).open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    delimiter = "\t" if "\t" in header else ","
    cols = {c.strip().lower() for c in header.strip().split(delimiter)}
    return {"speaker", "value"}.issubset(cols)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_plan() -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sources = discover_sources()

    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    missing_sources: list[int] = []
    transcript_ok: list[int] = []
    transcript_missing_or_bad: list[dict[str, object]] = []
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}

    for pid in EXPECTED_IDS:
        source_entry = sources[pid]
        chosen = choose_source(pid, source_entry)
        source_rows.append(
            {
                "pid": pid,
                "chosen_kind": chosen[0] if chosen else "",
                "chosen_path": str(chosen[1]) if chosen else "",
                "zip_candidates": "; ".join(str(p) for p in source_entry["zips"]),
                "folder_candidates": "; ".join(str(p) for p in source_entry["folders"]),
            }
        )
        if chosen is None:
            missing_sources.append(pid)
            continue

        kind, path = chosen
        edaic_map = edaic_files_for(pid)
        try:
            source_files = source_files_for(pid, kind, path)
        except Exception as exc:
            transcript_missing_or_bad.append({"pid": pid, "problem": f"cannot read source: {exc!r}", "source": str(path)})
            continue

        saw_transcript = False
        for sf in source_files:
            category = classify_file(sf, edaic_map)
            counts[category] = counts.get(category, 0) + 1
            totals[category] = totals.get(category, 0) + sf.size
            edaic_hit = edaic_map.get(sf.basename.lower())
            rows.append(
                {
                    "pid": pid,
                    "basename": sf.basename,
                    "source_kind": kind,
                    "source_path": str(path),
                    "member_name": sf.member_name,
                    "source_size": sf.size,
                    "category": category,
                    "edaic_path_same_windows_name": edaic_hit["path"] if edaic_hit else "",
                    "edaic_size_same_windows_name": edaic_hit["size"] if edaic_hit else "",
                }
            )

            if re.fullmatch(rf"{pid}_TRANSCRIPT\.csv", sf.basename, flags=re.IGNORECASE):
                saw_transcript = True
                try:
                    ok = (
                        has_speaker_columns_from_zip(path, sf.member_name)
                        if kind == "zip"
                        else has_speaker_columns_from_folder(path, sf.member_name)
                    )
                except Exception as exc:
                    transcript_missing_or_bad.append(
                        {"pid": pid, "problem": f"transcript read error: {exc!r}", "source": str(path)}
                    )
                else:
                    if ok:
                        transcript_ok.append(pid)
                    else:
                        transcript_missing_or_bad.append(
                            {"pid": pid, "problem": "transcript lacks speaker/value columns", "source": str(path)}
                        )

        if not saw_transcript:
            transcript_missing_or_bad.append({"pid": pid, "problem": "no *_TRANSCRIPT.csv in source", "source": str(path)})

    write_csv(rows, OUTDIR / "file_classification.csv")
    write_csv(source_rows, OUTDIR / "source_candidates.csv")
    write_csv(transcript_missing_or_bad, OUTDIR / "transcript_problems.csv")

    summary = {
        "expected_ids": len(EXPECTED_IDS),
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources,
        "transcript_ok_count": len(set(transcript_ok)),
        "transcript_problem_count": len(transcript_missing_or_bad),
        "counts": counts,
        "bytes": totals,
        "human_bytes": {k: human_size(v) for k, v in totals.items()},
        "interpretation": {
            "safe_minimal_extract": "Extract only original DAIC-WOZ speaker transcripts to a separate folder under E-daic.",
            "why_not_same_participant_folder": "Windows treats *_Transcript.csv and *_TRANSCRIPT.csv as the same filename, but eDAIC and DAIC-WOZ contents differ.",
            "why_not_all_nonduplicates_to_f": "Most DAIC-only optional files are large CLNF/COVAREP/FORMANT feature files and F: has too little free space.",
        },
        "output_files": {
            "classification_csv": str(OUTDIR / "file_classification.csv"),
            "sources_csv": str(OUTDIR / "source_candidates.csv"),
            "transcript_problems_csv": str(OUTDIR / "transcript_problems.csv"),
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# DAIC-WOZ 与 eDAIC 去重合并计划",
        "",
        f"- 期望 DAIC-WOZ participant 数: {len(EXPECTED_IDS)}",
        f"- 找不到源压缩包/文件夹: {len(missing_sources)}",
        f"- 可用原始 speaker transcript: {len(set(transcript_ok))}",
        "",
        "## 分类体积",
        "",
    ]
    for key in sorted(totals):
        md.append(f"- `{key}`: {counts.get(key, 0)} files, {human_size(totals[key])}")
    md.extend(
        [
            "",
            "## 建议",
            "",
            "不要整包解压到 F:。音频大概率已在 eDAIC 中重复，原始 DAIC-WOZ 的 CLNF/COVAREP/FORMANT 等特征虽不是同名重复，但体积很大，而且当前研究首先需要的是带 `speaker` 的 `*_TRANSCRIPT.csv`。",
            "",
            "最安全做法：只抽取 `*_TRANSCRIPT.csv` 到 `F:\\数据库\\E-daic\\DAIC-WOZ_original_transcripts`，不要放进每个 `*_P.tar\\*_P` 目录，因为 Windows 下 `*_TRANSCRIPT.csv` 会和 eDAIC 的 `*_Transcript.csv` 发生同名冲突。",
        ]
    )
    (OUTDIR / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_transcripts(dest: Path) -> dict[str, object]:
    dest.mkdir(parents=True, exist_ok=True)
    sources = discover_sources()
    extracted: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    for pid in EXPECTED_IDS:
        chosen = choose_source(pid, sources[pid])
        if chosen is None:
            problems.append({"pid": pid, "problem": "missing source"})
            continue
        kind, path = chosen
        target = dest / f"{pid}_TRANSCRIPT.csv"
        try:
            if kind == "folder":
                source_file = path / f"{pid}_TRANSCRIPT.csv"
                if not source_file.exists():
                    problems.append({"pid": pid, "problem": "missing transcript in folder", "source": str(path)})
                    continue
                data = source_file.read_bytes()
            else:
                with zipfile.ZipFile(path) as zf:
                    matches = [
                        info
                        for info in zf.infolist()
                        if not info.is_dir()
                        and re.search(rf"(^|/){pid}_TRANSCRIPT\.csv$", info.filename, flags=re.IGNORECASE)
                    ]
                    if not matches:
                        problems.append({"pid": pid, "problem": "missing transcript in zip", "source": str(path)})
                        continue
                    with zf.open(matches[0]) as f:
                        data = f.read()
            target.write_bytes(data)
            extracted.append(
                {
                    "pid": pid,
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        except Exception as exc:
            problems.append({"pid": pid, "problem": repr(exc), "source": str(path)})

    write_csv(extracted, OUTDIR / "extracted_transcripts.csv")
    write_csv(problems, OUTDIR / "extract_problems.csv")
    result = {
        "destination": str(dest),
        "extracted_count": len(extracted),
        "problem_count": len(problems),
        "total_bytes": sum(int(row["bytes"]) for row in extracted),
        "human_total_bytes": human_size(sum(int(row["bytes"]) for row in extracted)),
        "problems": problems[:20],
    }
    (OUTDIR / "extract_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-transcripts", action="store_true")
    parser.add_argument(
        "--dest",
        default=str(EDAIC_ROOT / "DAIC-WOZ_original_transcripts"),
        help="Destination used with --extract-transcripts.",
    )
    args = parser.parse_args()

    summary = build_plan()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.extract_transcripts:
        result = extract_transcripts(Path(args.dest))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
