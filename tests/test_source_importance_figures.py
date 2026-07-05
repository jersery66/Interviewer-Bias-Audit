from __future__ import annotations

from pathlib import Path
import subprocess


def test_build_source_importance_figures_from_strict_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    analysis_root = repo_root / "analysis_v2"
    python_exe = Path(r"E:\python3.12.8\python.exe")
    code = (
        "from reanalysis_v2.source_importance_figures import build_source_importance_figures; "
        f"build_source_importance_figures(r'{analysis_root}', r'{tmp_path}')"
    )

    result = subprocess.run(
        [str(python_exe), "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for stem in (
        "figure_1_source_decomposition_framework",
        "figure_2_incremental_model_auc",
        "figure_3_permutation_importance",
        "figure_4_domain_leave_one_out",
    ):
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            assert (tmp_path / f"{stem}{suffix}").stat().st_size > 0

    figure_2 = (tmp_path / "figure_2_incremental_model_auc.svg").read_text(
        encoding="utf-8"
    )
    figure_3 = (tmp_path / "figure_3_permutation_importance.svg").read_text(
        encoding="utf-8"
    )
    assert "BH-FDR" in figure_2
    assert "q=" in figure_2
    assert "**" in figure_3
    assert "q=0.0039" in figure_3
