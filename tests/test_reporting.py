from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from reanalysis_v2.reporting import (
    _inventory_audit,
    _strict_audit,
    frozen_text_hash_match,
    c5_plot_y_layout,
    build_final_package,
    filter_significant,
    verify_inventory,
    optional_upstream_hash_status,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_frozen_text_hash_accepts_only_cross_platform_newlines(tmp_path: Path) -> None:
    path = tmp_path / "frozen.csv"
    lf_bytes = b"a,b\n1,2\n"
    path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    expected = hashlib.sha256(lf_bytes).hexdigest()

    matched, observed = frozen_text_hash_match(path, expected)
    assert matched is True
    assert observed["canonical_lf_sha256"] == expected

    path.write_bytes(b"a,b\r\n1,3\r\n")
    matched, _ = frozen_text_hash_match(path, expected)
    assert matched is False


def test_verify_inventory_detects_ok_missing_and_hash_mismatch(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("stable", encoding="utf-8")
    bad.write_text("changed", encoding="utf-8")
    inventory = pd.DataFrame(
        [
            {
                "relative_path": "good.txt",
                "bytes": good.stat().st_size,
                "sha256": _sha256(good),
            },
            {
                "relative_path": "bad.txt",
                "bytes": 3,
                "sha256": hashlib.sha256(b"old").hexdigest(),
            },
            {
                "relative_path": "missing.txt",
                "bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            },
        ]
    )
    manifest = tmp_path / "output_manifest_sha256.csv"
    inventory.to_csv(manifest, index=False)

    result = verify_inventory(manifest)

    assert result.set_index("relative_path")["status"].to_dict() == {
        "good.txt": "PASS",
        "bad.txt": "HASH_MISMATCH",
        "missing.txt": "MISSING",
    }


def test_verify_inventory_accepts_git_crlf_checkout_for_text_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "frozen.csv"
    lf_bytes = b"participant_id,value\n9002,1\n"
    artifact.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    manifest = tmp_path / "output_manifest_sha256.csv"
    pd.DataFrame(
        [
            {
                "relative_path": artifact.name,
                "bytes": len(lf_bytes),
                "sha256": hashlib.sha256(lf_bytes).hexdigest(),
            }
        ]
    ).to_csv(manifest, index=False)

    result = verify_inventory(manifest)

    assert result.loc[0, "status"] == "PASS"


def test_verify_inventory_accepts_git_lf_checkout_for_crlf_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "frozen.csv"
    lf_bytes = b"participant_id,value\n9002,1\n"
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    artifact.write_bytes(lf_bytes)
    manifest = tmp_path / "output_manifest_sha256.csv"
    pd.DataFrame(
        [
            {
                "relative_path": artifact.name,
                "bytes": len(crlf_bytes),
                "sha256": hashlib.sha256(crlf_bytes).hexdigest(),
            }
        ]
    ).to_csv(manifest, index=False)

    result = verify_inventory(manifest)

    assert result.loc[0, "status"] == "PASS"


def test_distributed_inventory_audit_excludes_regenerable_embedding_caches() -> None:
    analysis_root = Path(__file__).resolve().parents[1] / "analysis_v2"

    audit = _inventory_audit(analysis_root)

    assert not audit["relative_path"].str.contains("embedding_cache/", regex=False).any()
    assert audit["status"].eq("PASS").all()


def test_strict_audit_accepts_relocated_frozen_turn_source() -> None:
    analysis_root = Path(__file__).resolve().parents[1] / "analysis_v2"

    audit = _strict_audit(analysis_root).set_index("check")

    assert audit.loc["c4:run_used_frozen_turn_source", "status"] == "PASS"


def test_missing_external_upstream_is_not_a_public_integrity_failure(tmp_path: Path) -> None:
    status = optional_upstream_hash_status(
        tmp_path / "restricted-upstream.csv",
        "expected-private-hash",
        frozen_copy_verified=True,
    )

    assert status["passed"] is True
    assert status["availability"] == "unavailable"
    assert "frozen copy verified" in status["observed"]


def test_reporting_uses_headless_matplotlib_backend() -> None:
    import matplotlib as mpl

    assert mpl.get_backend().lower() == "agg"


def test_filter_significant_uses_bh_q_and_keeps_direction() -> None:
    tests = pd.DataFrame(
        {
            "condition_a": ["a", "c"],
            "condition_b": ["b", "d"],
            "observed_difference": [0.2, -0.1],
            "fdr_bh_q": [0.049, 0.05],
        }
    )

    result = filter_significant(tests)

    assert result["condition_a"].tolist() == ["a"]
    assert result["direction"].tolist() == ["a > b"]


def test_c5_plot_layout_keeps_category_labels_separated() -> None:
    layout = c5_plot_y_layout(5)

    ordered = [layout["random"], layout["reviewed"], *layout["controls"]]
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    assert min(gaps) >= 0.8


def test_build_final_package_from_frozen_outputs(tmp_path: Path) -> None:
    analysis_root = Path(__file__).resolve().parents[1] / "analysis_v2"
    output = tmp_path / "06_tables_figures"

    result = build_final_package(analysis_root, output)

    assert result["status"] == "complete"
    assert result["strict_audit_passed"] is True
    assert result["inventory_failures"] == 0
    audit = pd.read_csv(output / "strict_contract_audit.csv").set_index("check")
    assert audit.loc["c4:frozen_turn_source_hash", "status"] == "PASS"
    assert audit.loc["c4:turn_alignment_c3", "status"] == "PASS"
    assert audit.loc["c4:turn_alignment_c4", "status"] == "PASS"
    assert audit.loc["c4:run_used_frozen_turn_source", "status"] == "PASS"
    assert len(pd.read_csv(output / "main_text" / "table_1_tfidf_primary.csv")) == 5
    assert len(pd.read_csv(output / "main_text" / "table_2_representation_robustness.csv")) == 15
    assert len(pd.read_csv(output / "supplement" / "table_s4_c5_controls.csv")) == 16
    assert len(pd.read_csv(output / "supplement" / "table_s6_c4_controls.csv")) == 5
    for stem in (
        output / "main_text" / "figure_1_representation_grid",
        output / "main_text" / "figure_2_threshold_calibration",
        output / "supplement" / "figure_s1_c5_controls",
        output / "supplement" / "figure_s2_c4_controls",
    ):
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            assert stem.with_suffix(suffix).stat().st_size > 0
    validation = (output / "validation_report.md").read_text(encoding="utf-8")
    assert "Verification Status: ANALYZED" in validation
    assert "11/11" in validation
    assert "independent external validation" not in validation.lower()
    reproducibility = (output / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "run_embedding_model.py --model mpnet" in reproducibility
    assert "run_embedding_model.py --model bge" in reproducibility
    assert "--model all-mpnet-base-v2" not in reproducibility
    assert "--model bge-large-en-v1.5" not in reproducibility
    package_inventory = verify_inventory(output / "output_manifest_sha256.csv")
    assert package_inventory["status"].eq("PASS").all()


def test_build_final_package_script_exposes_cli_help() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_final_package.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--analysis-root" in result.stdout
    assert "--output" in result.stdout
