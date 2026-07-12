from __future__ import annotations

import json
from pathlib import Path

from scripts.record_recomputation import compare_recomputations


def test_compare_recomputations_records_normalized_equal_csv_hashes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "metric.csv").write_bytes(b"a,b\n1,2\n")
    (second / "metric.csv").write_bytes(b"a,b\r\n1,2\r\n")

    report = compare_recomputations(first, second, "run-1", "run-2")

    assert report["status"] == "PASS"
    assert report["file_count"] == 1
    assert report["files"][0]["equal"] is True

