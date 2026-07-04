from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    package_root = Path(__file__).resolve().parents[1]
    frozen_result = (
        package_root
        / "analysis_v2"
        / "02_tfidf_main"
        / "metrics"
        / "primary_metrics.csv"
    )
    if frozen_result.exists():
        return
    marker = pytest.mark.skip(
        reason="完整结果未随脚本快照复制；生成 analysis_v2 全结果后自动启用"
    )
    for item in items:
        if item.name == "test_build_final_package_from_frozen_outputs":
            item.add_marker(marker)
