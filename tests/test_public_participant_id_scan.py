from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER = REPO_ROOT / "scripts" / "check_public_participant_ids.py"


def _init_git_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)


def _write_roster(path: Path) -> None:
    path.write_text(
        "participant_id\n9101\n9102\n9103\n9104\n9105\n",
        encoding="utf-8",
    )


def _run_scan(repo: Path, roster: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            "--repo",
            str(repo),
            "--roster",
            str(roster),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_roster_scan_fails_on_identifier_contexts_in_tracked_text(tmp_path: Path) -> None:
    repo = tmp_path / "public"
    roster = tmp_path / "restricted_roster.csv"
    _write_roster(roster)
    _init_git_repo(
        repo,
        {
            "README.md": (
                "ID 9102; participant_9101; case 9105; subject 9103.\n"
                "Allowed aggregates: n=9103 and a frozen 9104-cell consensus.\n"
            ),
            "data.csv": "participant_id,value\n9101,1\n",
            "config.json": '{"participant_id": 9105, "bootstrap": 9103}\n',
            "tests/example.py": (
                "ids = np.arange(9103, 9103 + n)\n"
                'assert frame.loc[frame["participant_id"] == 9105].empty\n'
            ),
            "case_9102.md": "opaque role description only\n",
        },
    )

    result = _run_scan(repo, roster)

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "FAIL"
    assert report["roster_id_count"] == 5
    assert report["match_count"] == 10
    assert {match["reason"] for match in report["matches"]} == {
        "identifier_context",
        "identifier_field",
        "identifier_path",
    }


def test_roster_scan_allows_aggregate_counts_hashes_decimals_and_parameters(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "public"
    roster = tmp_path / "restricted_roster.csv"
    _write_roster(roster)
    _init_git_repo(
        repo,
        {
            "README.md": "n=9103; 9104-cell consensus; value=0.9102; year=2026\n",
            "results.csv": "metric,n,value\nauc,9103,0.9102\n",
            "manifest.json": '{"sha256": "abc9102def9103abc", "bootstrap": 9103}\n',
            "test.py": "n_permutations=9103  # noqa: E402\n",
            "figure.svg": '<svg><path id="font-glyph" d="M 9103 2009"/></svg>\n',
        },
    )

    result = _run_scan(repo, roster)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["match_count"] == 0


def test_roster_scan_rejects_a_roster_inside_the_public_repo(tmp_path: Path) -> None:
    repo = tmp_path / "public"
    _init_git_repo(repo, {"README.md": "safe\n"})
    roster = repo / "roster.csv"
    _write_roster(roster)

    result = _run_scan(repo, roster)

    assert result.returncode == 2
    assert "outside the public repository" in result.stderr


def test_packet_builder_requires_the_restricted_sentinel_id_at_runtime() -> None:
    from scripts.prepare_c5_blind_audit_v3 import prepare

    parameters = inspect.signature(prepare).parameters

    assert "sentinel_participant_id" in parameters
    assert parameters["sentinel_participant_id"].default is inspect.Parameter.empty
