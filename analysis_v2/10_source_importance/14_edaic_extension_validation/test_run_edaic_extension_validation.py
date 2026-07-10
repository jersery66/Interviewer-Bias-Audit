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


if __name__ == "__main__":
    unittest.main()
