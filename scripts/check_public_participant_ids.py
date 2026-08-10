"""Fail a public release when tracked text exposes restricted participant IDs.

The official roster is supplied at runtime from outside the public repository.
No roster values or crosswalk are stored by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
BINARY_EXTENSIONS = {
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".tif",
    ".tiff",
    ".ttf",
    ".woff",
    ".xlsx",
    ".zip",
}
IDENTIFIER_CONTEXT = re.compile(
    r"(?i)(?<![A-Za-z])(?:participant(?:[_ -]?id)?|subject(?:[_ -]?id)?|"
    r"case(?:[_ -]?id)?|sentinel|ids?)(?![A-Za-z])|哨兵|个案|被试|参与者"
)
IDENTIFIER_FIELD_NAMES = {
    "id",
    "case_id",
    "participant",
    "participant_id",
    "participantid",
    "subject",
    "subject_id",
    "subjectid",
}
LONG_HEX = re.compile(r"[A-Fa-f0-9]{8,}")
GROUPED_INTEGER = re.compile(r"(?<![A-Za-z0-9])\d[\d_]*\d(?![A-Za-z0-9])")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_roster(path: Path, id_column: str) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(f"roster is missing required column: {id_column}")
        values = [str(row[id_column]).strip() for row in reader]
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("roster participant IDs must be non-empty decimal integers")
    if len(values) != len(set(values)):
        raise ValueError("roster participant IDs must be unique")
    return tuple(sorted(values, key=lambda value: (-len(value), value)))


def _tracked_files(repo: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo,
    )
    return [value for value in output.decode("utf-8").split("\0") if value]


def _decode_searchable_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return None
    data = path.read_bytes()
    if b"\x00" in data[:8192] and path.suffix.lower() not in TEXT_EXTENSIONS:
        return None
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _inside_long_hex(line: str, start: int, end: int) -> bool:
    return any(left <= start and end <= right for left, right in (m.span() for m in LONG_HEX.finditer(line)))


def _inside_larger_grouped_integer(line: str, start: int, end: int, value: str) -> bool:
    for match in GROUPED_INTEGER.finditer(line):
        if match.start() <= start and end <= match.end():
            return match.group(0).replace("_", "") != value
    return False


def _near_identifier_context(
    line: str,
    start: int,
    end: int,
    *,
    ignore_xml_id_attributes: bool = False,
) -> bool:
    for match in IDENTIFIER_CONTEXT.finditer(line):
        if (
            ignore_xml_id_attributes
            and match.group(0).lower() == "id"
            and line[match.end() :].lstrip().startswith("=")
        ):
            continue
        distance = max(match.start() - end, start - match.end(), 0)
        if distance <= 48:
            return True
    return False


def _candidate_pattern(roster: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(value) for value in roster)
    return re.compile(rf"(?<!\d)(?P<participant_id>{alternatives})(?!\d)")


def _match_dict(
    *,
    path: str,
    participant_id: str,
    reason: str,
    line: int | None,
    excerpt: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "line": line,
        "participant_id": participant_id,
        "reason": reason,
        "excerpt": excerpt[:240],
    }


def _scan_csv_identifier_fields(
    path: Path,
    relative: str,
    roster: set[str],
    line_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            fields = [
                field
                for field in reader.fieldnames
                if field.strip().lower() in IDENTIFIER_FIELD_NAMES
            ]
            if not fields:
                return []
            existing = {
                (match["participant_id"], match["line"])
                for match in line_matches
                if match["path"] == relative
            }
            matches: list[dict[str, Any]] = []
            for row_number, row in enumerate(reader, start=2):
                for field in fields:
                    value = str(row.get(field, "")).strip()
                    if value in roster and (value, row_number) not in existing:
                        matches.append(
                            _match_dict(
                                path=relative,
                                line=row_number,
                                participant_id=value,
                                reason="identifier_field",
                                excerpt=f"{field}=<restricted participant ID>",
                            )
                        )
            return matches
    except (csv.Error, UnicodeDecodeError):
        return []


def scan_repository(repo: Path, roster: tuple[str, ...]) -> dict[str, Any]:
    tracked = _tracked_files(repo)
    candidate = _candidate_pattern(roster)
    roster_set = set(roster)
    matches: list[dict[str, Any]] = []
    text_files_scanned = 0

    for relative in tracked:
        path = repo / relative
        path_text = relative.replace("\\", "/")
        for match in candidate.finditer(path_text):
            value = match.group("participant_id")
            if _near_identifier_context(path_text, match.start(), match.end()):
                matches.append(
                    _match_dict(
                        path=relative,
                        line=None,
                        participant_id=value,
                        reason="identifier_path",
                        excerpt=path_text,
                    )
                )

        text = _decode_searchable_text(path)
        if text is None:
            continue
        text_files_scanned += 1
        line_matches: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in candidate.finditer(line):
                value = match.group("participant_id")
                if (match.start() and line[match.start() - 1] == ".") or (
                    match.end() < len(line) and line[match.end()] == "."
                ):
                    continue
                if _inside_long_hex(line, match.start(), match.end()):
                    continue
                if _inside_larger_grouped_integer(
                    line, match.start(), match.end(), value
                ):
                    continue
                if not _near_identifier_context(
                    line,
                    match.start(),
                    match.end(),
                    ignore_xml_id_attributes=path.suffix.lower()
                    in {".html", ".svg", ".xml"},
                ):
                    continue
                line_matches.append(
                    _match_dict(
                        path=relative,
                        line=line_number,
                        participant_id=value,
                        reason="identifier_context",
                        excerpt=line.strip(),
                    )
                )
        matches.extend(line_matches)
        if path.suffix.lower() == ".csv":
            matches.extend(
                _scan_csv_identifier_fields(
                    path,
                    relative,
                    roster_set,
                    line_matches,
                )
            )

    matches.sort(
        key=lambda item: (
            item["path"],
            -1 if item["line"] is None else item["line"],
            item["participant_id"],
            item["reason"],
        )
    )
    return {
        "status": "FAIL" if matches else "PASS",
        "tracked_file_count": len(tracked),
        "text_file_count": text_files_scanned,
        "roster_id_count": len(roster),
        "match_count": len(matches),
        "matches": matches,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Git-tracked public text against a restricted participant roster."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--id-column", default="participant_id")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        roster_path = args.roster.resolve(strict=True)
        if not (repo / ".git").exists():
            raise ValueError(f"not a Git repository: {repo}")
        if _is_within(roster_path, repo):
            raise ValueError("restricted roster must remain outside the public repository")
        roster = _load_roster(roster_path, args.id_column)
        report = scan_repository(repo, roster)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"participant-ID scan configuration error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"{report['status']}: scanned {report['tracked_file_count']} tracked files "
            f"against {report['roster_id_count']} restricted participant IDs; "
            f"matches={report['match_count']}"
        )
        for match in report["matches"]:
            location = match["path"]
            if match["line"] is not None:
                location += f":{match['line']}"
            print(f"{location}: {match['reason']}: {match['participant_id']}")
    return 1 if report["matches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
