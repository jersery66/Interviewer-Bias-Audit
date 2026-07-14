from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "analysis_v2" / "12_submission_audit"
FORMAL_IDENTIFICATION_DIR = (
    ROOT
    / "analysis_v2"
    / "10_source_importance"
    / "08_identification_sensitivity"
    / "run_formal_10x"
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".svg", ".tsv", ".txt", ".yaml", ".yml"}
ACCEPTED_PLAN_STATUSES = {
    "frozen_before_sensitivity_results",
    "original_plan_frozen_before_sensitivity_results",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_candidates(path: Path) -> set[str]:
    raw = path.read_bytes()
    candidates = {hashlib.sha256(raw).hexdigest()}
    if path.suffix.lower() in TEXT_SUFFIXES:
        candidates.add(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
    return candidates


def plan_status_is_acceptable(status: str) -> bool:
    if status in ACCEPTED_PLAN_STATUSES:
        return True
    return status.startswith("original_plan_frozen_before_sensitivity_results;") and (
        "closeout_amendment_frozen_before_closeout_inference_rerun" in status
    )


def verify_formal_identification_output(formal_dir: Path) -> list[str]:
    """Verify the committed, derived 10-repeat pairing/Fake-D result package.

    This is a result-integrity gate only. It does not rerun the restricted-data
    analysis and therefore deliberately checks the frozen manifest, output
    hashes, schema, row counts, and the no-self-match invariant instead.
    """

    errors: list[str] = []
    manifest_path = formal_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return ["formal identification manifest missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"formal identification manifest invalid: {exc}"]

    if manifest.get("status") != "complete":
        errors.append("formal identification status is not complete")
    if manifest.get("run_class") != "formal":
        errors.append("formal identification run_class is not formal")
    if manifest.get("fusion_method") != "early_concat":
        errors.append("formal identification fusion method is not early_concat")

    repeats = manifest.get("repeats")
    if repeats != list(range(1, 11)):
        errors.append("formal identification repeats are not 1..10")
    if manifest.get("pairing_draws_per_repeat") != 50:
        errors.append("formal pairing draw count is not 50")
    if manifest.get("fake_d_draws_per_repeat") != 50:
        errors.append("formal Fake-D draw count is not 50")
    if manifest.get("embedding_models") != [
        "model_1_all_mpnet_base_v2",
        "model_2_bge_large_en_v1_5",
    ]:
        errors.append("formal identification embeddings are not the locked MPNet/BGE pair")
    if manifest.get("primary_contrasts") != [
        "real_minus_matched",
        "real_minus_random",
        "d_delta_minus_fake",
    ]:
        errors.append("formal identification primary contrast contract changed")
    if not str(manifest.get("code_commit", "")).strip():
        errors.append("formal identification code commit missing")

    output_hashes = manifest.get("output_file_hashes", {})
    expected_outputs = {
        "pairing_draw_metrics.csv",
        "pairing_oof_predictions.csv",
        "pairing_ledger.csv",
        "fake_d_draw_metrics.csv",
        "fake_d_oof_predictions.csv",
        "fake_d_ledger.csv",
        "repeat_contrasts.csv",
        "primary_contrasts.csv",
    }
    if set(output_hashes) != expected_outputs:
        errors.append("formal identification output hash inventory changed")
    for name in sorted(expected_outputs):
        path = formal_dir / name
        if not path.is_file():
            errors.append(f"formal identification output missing: {name}")
        elif str(output_hashes.get(name, "")) not in hash_candidates(path):
            errors.append(f"formal identification output hash mismatch: {name}")

    def read_csv(name: str) -> pd.DataFrame | None:
        path = formal_dir / name
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"formal identification CSV invalid ({name}): {exc}")
            return None

    primary = read_csv("primary_contrasts.csv")
    if primary is not None:
        required_columns = {
            "embedding_model",
            "contrast",
            "n_repeats",
            "mean",
            "ci_low",
            "ci_high",
            "raw_p",
            "bh_q",
        }
        if not required_columns.issubset(primary.columns):
            errors.append("formal primary contrast columns incomplete")
        else:
            if len(primary) != 6:
                errors.append("formal primary contrast row count is not six")
            if set(primary["embedding_model"]) != set(manifest.get("embedding_models", [])):
                errors.append("formal primary contrast embedding set changed")
            if set(primary["contrast"]) != set(manifest.get("primary_contrasts", [])):
                errors.append("formal primary contrast set changed")
            numeric = primary[
                ["n_repeats", "mean", "ci_low", "ci_high", "raw_p", "bh_q"]
            ].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any() or not numeric.map(lambda value: value == value and abs(value) != float("inf")).all().all():
                errors.append("formal primary contrast values contain non-finite values")
            if not primary["n_repeats"].eq(10).all():
                errors.append("formal primary contrast repeat count is not ten")
            if not primary["raw_p"].between(0, 1).all() or not primary["bh_q"].between(0, 1).all():
                errors.append("formal primary contrast p/q outside [0, 1]")

    pairing_metrics = read_csv("pairing_draw_metrics.csv")
    fake_metrics = read_csv("fake_d_draw_metrics.csv")
    if pairing_metrics is not None:
        expected_rows = 2 * 10 * (1 + 2 * 50)
        if len(pairing_metrics) != expected_rows:
            errors.append(f"formal pairing metric row count is not {expected_rows}")
        counts = pairing_metrics.groupby(["embedding_model", "repeat", "condition"]).size()
        expected_counts = {
            "real": 1,
            "random": 50,
            "matched": 50,
        }
        for (embedding, repeat, condition), count in counts.items():
            if count != expected_counts.get(condition, -1):
                errors.append(f"formal pairing draw count mismatch: {embedding}/{repeat}/{condition}")
    if fake_metrics is not None:
        expected_rows = 2 * 10 * (1 + 50)
        if len(fake_metrics) != expected_rows:
            errors.append(f"formal Fake-D metric row count is not {expected_rows}")
        counts = fake_metrics.groupby(["embedding_model", "repeat", "condition"]).size()
        for (embedding, repeat, condition), count in counts.items():
            if condition == "real" and count != 1:
                errors.append(f"formal Fake-D real count mismatch: {embedding}/{repeat}")
            elif condition == "fake_d" and count != 50:
                errors.append(f"formal Fake-D draw count mismatch: {embedding}/{repeat}")

    for name in ("pairing_ledger.csv", "fake_d_ledger.csv"):
        ledger = read_csv(name)
        if ledger is not None and {"condition", "self_match"}.issubset(ledger.columns):
            non_real = ledger.loc[ledger["condition"] != "real", "self_match"]
            if non_real.astype(bool).any():
                errors.append(f"formal ledger contains self matches: {name}")

    return errors


def verify() -> list[str]:
    errors: list[str] = []
    inventory_path = AUDIT_DIR / "output_manifest_sha256.csv"
    inventory = pd.read_csv(inventory_path)
    for row in inventory.itertuples(index=False):
        path = AUDIT_DIR / str(row.file)
        if not path.is_file():
            errors.append(f"missing: {row.file}")
        elif str(row.sha256) not in hash_candidates(path):
            errors.append(f"hash mismatch: {row.file}")

    run_manifest = json.loads((AUDIT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    plan = AUDIT_DIR / run_manifest["post_audit_plan"]["file"]
    if str(run_manifest["post_audit_plan"]["sha256"]) not in hash_candidates(plan):
        errors.append("post-audit plan hash mismatch")
    if not plan_status_is_acceptable(str(run_manifest["post_audit_plan"]["status"])):
        errors.append("post-audit plan was not recorded as frozen before results")
    if run_manifest["interpretation_guardrails"]["ci"] != (
        "code_tests_and_derived_integrity_not_full_restricted_data_reproduction"
    ):
        errors.append("CI interpretation boundary missing")

    trace = pd.read_csv(AUDIT_DIR / "c5_traceability_audit_no_quotes.csv")
    forbidden_columns = {"quote", "text", "source_text", "original_quote", "corrected_quote"}
    leaked = forbidden_columns.intersection(column.lower() for column in trace.columns)
    if leaked:
        errors.append(f"public C5 audit contains text columns: {sorted(leaked)}")

    required = {
        "locked_brier_skill_metrics.csv",
        "symmetric_half_min_paired_comparison.csv",
        "symmetric_position_paired_comparisons.csv",
        "c5_retained_alignment_paired_differences.csv",
        "model_configuration.md",
        "submission_audit_summary.md",
    }
    missing_required = sorted(name for name in required if not (AUDIT_DIR / name).is_file())
    if missing_required:
        errors.append(f"required outputs missing: {missing_required}")
    errors.extend(verify_formal_identification_output(FORMAL_IDENTIFICATION_DIR))
    return errors


def main() -> int:
    errors = verify()
    if errors:
        print("Result integrity verification: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Code/test reproducibility: run pytest in the public clone")
    print("Result integrity verification: PASS")
    print("Full data-to-result reproduction: RESTRICTED INPUTS REQUIRED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
