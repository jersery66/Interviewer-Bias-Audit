import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_edaic_extension_validation.py")
SPEC = importlib.util.spec_from_file_location("module14", MODULE_PATH)
module14 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module14)


class ChunkResumeTests(unittest.TestCase):
    def test_chunk_jobs_use_one_record_and_distinct_run_id_per_chunk(self):
        jobs = module14.plan_chunk_jobs(603, "x" * 7000, max_chars=3500, overlap=200)

        self.assertEqual(3, len(jobs))
        self.assertEqual(3, len({job["run_id"] for job in jobs}))
        self.assertEqual([1, 2, 3], [job["chunk_index"] for job in jobs])
        self.assertTrue(all(job["participant_id"] == "603" for job in jobs))
        self.assertTrue(all(len(job["text"]) <= 3500 for job in jobs))

    def test_chunk_api_succeeded_requires_api_ok_raw_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "chunk_raw.jsonl"
            raw_path.write_text(
                json.dumps({"participant_id": "603", "api_ok": False}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(module14.chunk_api_succeeded(raw_path))

            raw_path.write_text(
                json.dumps({"participant_id": "603", "api_ok": True}) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(module14.chunk_api_succeeded(raw_path))

    def test_resume_seed_marks_only_declared_chunks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_path = Path(tmp) / "resume_seed.json"
            seed_path.write_text(
                json.dumps({"completed_chunks": {"601": [1], "603": [4]}}),
                encoding="utf-8",
            )

            completed = module14.load_resume_completed_chunks(seed_path)

            self.assertEqual({("601", 1), ("603", 4)}, completed)
            self.assertNotIn(("603", 1), completed)

    def test_semantic_span_dedup_ignores_chunk_local_order(self):
        spans = pd.DataFrame([
            {
                "participant_id": 680,
                "domain": "mental_health_history",
                "polarity": "present",
                "exact_quote": "same evidence",
                "source_order": 1,
            },
            {
                "participant_id": 680,
                "domain": "mental_health_history",
                "polarity": "present",
                "exact_quote": "same evidence",
                "source_order": 7,
            },
            {
                "participant_id": 680,
                "domain": "depressed_mood",
                "polarity": "present",
                "exact_quote": "same evidence",
                "source_order": 2,
            },
        ])

        deduped = module14.deduplicate_spans(spans)

        self.assertEqual(2, len(deduped))
        self.assertEqual(
            {"mental_health_history", "depressed_mood"},
            set(deduped["domain"]),
        )

    def test_error_output_has_quadrant_rows_and_nonzero_exact_fisher_p(self):
        labels = np.array(([0, 1] * 11), dtype=int)
        quadrants = np.array(
            ["self_high_evidence_low"] * 4
            + ["self_low_evidence_high"] * 5
            + ["consistent_high"] * 6
            + ["consistent_low"] * 7,
            dtype=object,
        )
        mismatch = np.isin(quadrants, module14.MISMATCH_QUADRANTS)
        pred = labels.copy()
        pred[mismatch] = 1 - pred[mismatch]
        prob = np.where(pred == 1, 0.9, 0.1)
        transfer = pd.DataFrame({
            "label": labels,
            "prob_base": prob,
            "pred_base": pred,
            "prob_count": prob,
            "pred_count": pred,
            "prob_pres": prob,
            "pred_pres": pred,
            "_q_coverage_breadth_predefined_5plus": quadrants,
            "_q_evidence_density_predefined_11plus": quadrants,
        })

        original_dir = module14.SCRIPT_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                module14.SCRIPT_DIR = Path(tmp)
                _, errors = module14.compute_metrics_and_errors(transfer)
        finally:
            module14.SCRIPT_DIR = original_dir

        expected_groups = {
            "mismatch_vs_consistent",
            "mismatch",
            "consistent",
            *module14.QUADRANTS,
        }
        self.assertTrue({"n_correct", "n_error", "error_rate"}.issubset(errors.columns))
        self.assertEqual(expected_groups, set(errors["group"]))
        comparison = errors[
            (errors["model"] == "base")
            & (errors["evidence_def"] == "coverage_breadth")
            & (errors["group"] == "mismatch_vs_consistent")
        ].iloc[0]
        self.assertGreater(float(comparison["fisher_p"]), 0.0)
        self.assertLess(float(comparison["fisher_p"]), 0.0001)

    def test_transcript_validation_rejects_empty_and_missing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.csv"
            empty.write_bytes(b"")
            missing_text = root / "missing_text.csv"
            missing_text.write_text("Start_Time,End_Time\n0,1\n", encoding="utf-8")
            valid = root / "valid.csv"
            valid.write_text(
                "Start_Time,End_Time,Text,Confidence\n0,1,hello,0.9\n",
                encoding="utf-8",
            )

            self.assertEqual("transcript_empty", module14.transcript_exclusion_reason(empty))
            self.assertEqual(
                "transcript_format_missing_text",
                module14.transcript_exclusion_reason(missing_text),
            )
            self.assertEqual("", module14.transcript_exclusion_reason(valid))

    def test_edaic_transcript_path_is_portable_and_resolvable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "data" / "data" / "601_P" / "601_Transcript.csv"
            transcript.parent.mkdir(parents=True)
            transcript.write_text("Text\nhello\n", encoding="utf-8")

            relative = module14.portable_edaic_path(transcript, root)
            resolved = module14.resolve_edaic_path(relative, root)

            self.assertEqual("data/data/601_P/601_Transcript.csv", relative)
            self.assertEqual(transcript.resolve(), resolved.resolve())


class MethodologicalDowngradeTests(unittest.TestCase):
    @staticmethod
    def make_transfer():
        labels = np.array([1] * 6 + [0] * 16, dtype=int)
        probabilities = {
            "base": np.array([
                0.90, 0.76, 0.61, 0.42, 0.31, 0.18,
                0.84, 0.72, 0.66, 0.55, 0.48, 0.39, 0.34, 0.29,
                0.24, 0.20, 0.16, 0.13, 0.10, 0.08, 0.05, 0.02,
            ]),
            "count": np.array([
                0.82, 0.69, 0.54, 0.45, 0.28, 0.16,
                0.88, 0.75, 0.63, 0.58, 0.51, 0.41, 0.36, 0.30,
                0.25, 0.21, 0.17, 0.12, 0.09, 0.07, 0.04, 0.01,
            ]),
            "pres": np.array([
                0.79, 0.67, 0.52, 0.43, 0.27, 0.15,
                0.86, 0.73, 0.62, 0.57, 0.50, 0.40, 0.35, 0.31,
                0.26, 0.22, 0.18, 0.14, 0.11, 0.06, 0.03, 0.01,
            ]),
        }
        quadrants = np.array(
            ["self_high_evidence_low"] * 4
            + ["self_low_evidence_high"] * 5
            + ["consistent_high"] * 6
            + ["consistent_low"] * 7,
            dtype=object,
        )
        transfer = pd.DataFrame({
            "participant_id": np.arange(601, 623),
            "label": labels,
            "phq8_score": np.where(labels == 1, 12, 4),
            "coverage_breadth": np.arange(22) % 10,
            "evidence_density": np.arange(22) % 18,
            "_q_coverage_breadth_predefined_5plus": quadrants,
            "_q_evidence_density_predefined_11plus": quadrants,
        })
        for model, prob in probabilities.items():
            transfer[f"prob_{model}"] = prob
            transfer[f"pred_{model}"] = (prob >= 0.5).astype(int)
        return transfer

    @staticmethod
    def make_quadrants():
        rows = []
        counts = {
            "self_high_evidence_low": 4,
            "self_low_evidence_high": 5,
            "consistent_high": 6,
            "consistent_low": 7,
        }
        for evidence_def, rule in [
            ("coverage_breadth", "predefined_5plus"),
            ("evidence_density", "predefined_11plus"),
        ]:
            for quadrant, n in counts.items():
                rows.append({
                    "evidence_def": evidence_def,
                    "threshold_rule": rule,
                    "quadrant": quadrant,
                    "n": n,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def make_errors():
        rows = []
        for evidence_def, rule in [
            ("coverage_breadth", "predefined_5plus"),
            ("evidence_density", "predefined_11plus"),
        ]:
            for model in ["base", "count", "pres"]:
                rows.append({
                    "model": model,
                    "evidence_def": evidence_def,
                    "threshold_rule": rule,
                    "group": "mismatch_vs_consistent",
                    "n_mismatch": 9,
                    "n_consistent": 13,
                    "n_error_mismatch": 7,
                    "n_error_consistent": 3,
                    "error_rate_mismatch": 7 / 9,
                    "error_rate_consistent": 3 / 13,
                })
        return pd.DataFrame(rows)

    def build_comparison_fixture(self, root):
        root = Path(root)
        coverage_path = root / "m12_coverage.csv"
        density_path = root / "m12_density.csv"
        m12_error_path = root / "m12_errors.csv"
        quadrants = self.make_quadrants()
        quadrants[quadrants["evidence_def"] == "coverage_breadth"].to_csv(
            coverage_path, index=False, encoding="utf-8-sig"
        )
        quadrants[quadrants["evidence_def"] == "evidence_density"].to_csv(
            density_path, index=False, encoding="utf-8-sig"
        )
        errors = self.make_errors()
        m12_errors = errors.copy()
        m12_errors["error_rate"] = np.nan
        m12_errors["tp"] = np.nan
        m12_errors["fn"] = np.nan
        m12_errors = pd.concat([
            m12_errors,
            pd.DataFrame([{
                "model": "base",
                "evidence_def": "overall",
                "threshold_rule": "overall",
                "group": "overall",
                "error_rate_mismatch": np.nan,
                "error_rate_consistent": np.nan,
                "error_rate": 0.25,
                "tp": 30,
                "fn": 13,
            }]),
        ], ignore_index=True)
        m12_errors.to_csv(m12_error_path, index=False, encoding="utf-8-sig")

        old_values = (
            module14.SCRIPT_DIR,
            module14.M12_QUAD_COV,
            module14.M12_QUAD_DEN,
            module14.M12_ERR,
        )
        try:
            module14.SCRIPT_DIR = root
            module14.M12_QUAD_COV = coverage_path
            module14.M12_QUAD_DEN = density_path
            module14.M12_ERR = m12_error_path
            return module14.build_comparison(
                self.make_transfer(),
                {"base": 0.72, "count": 0.74, "pres": 0.73},
                quadrants,
                errors,
            )
        finally:
            (
                module14.SCRIPT_DIR,
                module14.M12_QUAD_COV,
                module14.M12_QUAD_DEN,
                module14.M12_ERR,
            ) = old_values

    def test_metrics_report_auc_ci_and_mark_error_analysis_nonindependent(self):
        transfer = self.make_transfer()
        original_dir = module14.SCRIPT_DIR
        original_bootstrap_n = getattr(module14, "AUC_BOOTSTRAP_N", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                module14.SCRIPT_DIR = Path(tmp)
                module14.AUC_BOOTSTRAP_N = 500
                metrics, errors = module14.compute_metrics_and_errors(transfer)
        finally:
            module14.SCRIPT_DIR = original_dir
            if original_bootstrap_n is None:
                delattr(module14, "AUC_BOOTSTRAP_N")
            else:
                module14.AUC_BOOTSTRAP_N = original_bootstrap_n

        self.assertTrue({"AUC_CI95_low", "AUC_CI95_high"}.issubset(metrics.columns))
        self.assertTrue(metrics["AUC_CI95_low"].notna().all())
        self.assertTrue(metrics["AUC_CI95_high"].notna().all())
        for _, metric in metrics.iterrows():
            expected_low, expected_high = module14.stratified_bootstrap_auc_ci(
                transfer["label"],
                transfer[f"prob_{metric['model']}"],
                n_boot=500,
                seed=module14.RANDOM_SEED,
            )
            self.assertEqual(round(expected_low, 4), float(metric["AUC_CI95_low"]))
            self.assertEqual(round(expected_high, 4), float(metric["AUC_CI95_high"]))
        self.assertTrue((errors["independent_test"] == 0).all())
        comparison_notes = errors[errors["group"] == "mismatch_vs_consistent"]["note"]
        self.assertTrue(comparison_notes.str.contains("definition-feature overlap").all())

    def test_comparison_replaces_direction_binary_with_ci_n_and_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparison = self.build_comparison_fixture(tmp)

        self.assertNotIn("方向是否一致", comparison.columns)
        self.assertEqual(
            [
                "指标",
                "DAIC_WOZ_主分析",
                "E_DAIC_点估计",
                "E_DAIC_95CI",
                "E_DAIC_n",
                "限制说明",
            ],
            list(comparison.columns),
        )
        self.assertTrue(comparison["限制说明"].astype(str).str.len().gt(0).all())
        sample_row = comparison[comparison["指标"] == "样本量"].iloc[0]
        self.assertEqual("142", sample_row["DAIC_WOZ_主分析"])
        self.assertEqual("22", sample_row["E_DAIC_点估计"])
        auc_rows = comparison[comparison["指标"].str.contains("AUC")]
        self.assertTrue(auc_rows["E_DAIC_95CI"].astype(str).str.startswith("[").all())
        self.assertTrue(auc_rows["E_DAIC_n"].astype(str).str.contains("阳性=6").all())
        error_rows = comparison[comparison["指标"].str.contains("错分率")]
        self.assertTrue(error_rows["限制说明"].str.contains("定义—特征重叠").all())

    def test_summary_downgrades_error_concentration_to_nonindependent_description(self):
        transfer = self.make_transfer()
        quadrants = self.make_quadrants()
        errors = self.make_errors()
        manifest = pd.DataFrame({
            "included": [1] * len(transfer),
            "exclusion_reason": [""] * len(transfer),
        })
        with tempfile.TemporaryDirectory() as tmp:
            comparison = self.build_comparison_fixture(tmp)
        summary = module14.build_summary(
            manifest,
            transfer,
            quadrants,
            errors,
            {"base": 0.72, "count": 0.74, "pres": 0.73},
            comparison,
            4.5,
            10.5,
        )

        self.assertIn("定义—特征重叠", summary)
        self.assertIn("不能视为对错位误差机制的独立验证", summary)
        self.assertIn("95% CI", summary)
        self.assertNotIn("该方向与 DAIC-WOZ 主分析一致", summary)
        self.assertNotIn("方向上支持 DAIC-WOZ 主分析", summary)


if __name__ == "__main__":
    unittest.main()
