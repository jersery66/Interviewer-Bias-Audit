"""
DAIC-WOZ input-strategy experiment skeleton.

This script creates five text input conditions and runs minimal traditional
baselines. It is designed as a starting point; check DAIC-WOZ column names
before using it for final results.

Example:
python outputs/新研究_实验代码骨架.py ^
  --transcripts-dir path/to/DAIC-WOZ/transcripts ^
  --labels-csv path/to/labels.csv ^
  --split-csv path/to/split.csv ^
  --output-dir work/new_study_run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DIAGNOSTIC_PATTERNS = [
    r"\bdepress(ed|ion)?\b",
    r"\bsad\b",
    r"\bdown\b",
    r"\bhopeless\b",
    r"\bunhappy\b",
    r"\binterest\b",
    r"\bpleasure\b",
    r"\benjoy\b",
    r"\bsleep\b",
    r"\binsomnia\b",
    r"\btired\b",
    r"\bfatigue\b",
    r"\benergy\b",
    r"\bappetite\b",
    r"\beat(ing)?\b",
    r"\bweight\b",
    r"\bsuicid(e|al)\b",
    r"\bkill yourself\b",
    r"\bhurt yourself\b",
    r"\bself-harm\b",
    r"\bmental health\b",
    r"\btherapy\b",
    r"\bcounsel(or|lor)\b",
    r"\btreatment\b",
    r"\bmedication\b",
    r"\bantidepressant\b",
    r"\btrauma\b",
    r"\babuse\b",
    r"\bptsd\b",
    r"\bdiagnosed\b",
    r"\banxiety\b",
]

DIAGNOSTIC_RE = re.compile("|".join(DIAGNOSTIC_PATTERNS), flags=re.I)


def read_table(path: Path) -> pd.DataFrame:
    """Read DAIC-style CSV/TSV with a permissive separator fallback."""
    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) == 1:
            return pd.read_csv(path)
        return df
    except Exception:
        return pd.read_csv(path)


def normalize_speaker(raw: object) -> str:
    text = str(raw).strip().lower()
    if "ellie" in text or "interviewer" in text:
        return "interviewer"
    if "participant" in text:
        return "participant"
    return text or "unknown"


def infer_participant_id(path: Path) -> str:
    # Common DAIC-WOZ files look like 300_TRANSCRIPT.csv.
    return path.stem.split("_")[0]


def infer_label_columns(labels: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    id_candidates = ["participant_id", "Participant_ID", "Participant", "id", "ID"]
    phq_candidates = ["PHQ8_Score", "PHQ-8", "phq8", "phq8_score", "PHQ8"]
    binary_candidates = ["PHQ8_Binary", "depressed", "label", "target"]

    id_col = next((c for c in id_candidates if c in labels.columns), None)
    phq_col = next((c for c in phq_candidates if c in labels.columns), None)
    binary_col = next((c for c in binary_candidates if c in labels.columns), None)

    if id_col is None:
        raise ValueError(f"Cannot infer participant id column from {list(labels.columns)}")
    if phq_col is None and binary_col is None:
        raise ValueError("Need either PHQ-8 score column or binary label column.")
    return id_col, phq_col, binary_col


def prepare_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    id_col, phq_col, binary_col = infer_label_columns(labels)
    out = labels.copy()
    out["participant_id"] = out[id_col].astype(str)
    if binary_col is not None:
        out["label"] = out[binary_col].astype(int)
    else:
        out["label"] = (pd.to_numeric(out[phq_col], errors="coerce") >= 10).astype(int)
    return out[["participant_id", "label"]].drop_duplicates()


def read_transcript(path: Path) -> pd.DataFrame:
    df = read_table(path)
    lower_map = {c.lower(): c for c in df.columns}
    speaker_col = lower_map.get("speaker")
    value_col = lower_map.get("value") or lower_map.get("text") or lower_map.get("utterance")
    if speaker_col is None or value_col is None:
        raise ValueError(f"{path} must contain speaker and value/text/utterance columns.")

    out = df[[speaker_col, value_col]].copy()
    out.columns = ["speaker", "text"]
    out["speaker"] = out["speaker"].map(normalize_speaker)
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out = out[out["text"] != ""]
    return out


def join_lines(rows: Iterable[Tuple[str, str]], include_speaker: bool) -> str:
    lines = []
    for speaker, text in rows:
        if include_speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def load_evidence_summary(evidence_json_dir: Optional[Path], participant_id: str) -> str:
    if evidence_json_dir is None:
        return ""
    path = evidence_json_dir / f"{participant_id}.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines: List[str] = []
    if isinstance(data, dict):
        for category, quotes in data.items():
            if not quotes:
                continue
            if isinstance(quotes, str):
                quotes = [quotes]
            for quote in quotes:
                lines.append(f"{category}: {quote}")
    return "\n".join(lines)


def build_conditions(
    transcripts_dir: Path,
    evidence_json_dir: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, str]] = []
    stats: List[Dict[str, object]] = []

    transcript_files = sorted(transcripts_dir.glob("*TRANSCRIPT*.csv"))
    if not transcript_files:
        transcript_files = sorted(transcripts_dir.glob("*.csv"))

    for path in transcript_files:
        participant_id = infer_participant_id(path)
        df = read_transcript(path)

        full = join_lines(df[["speaker", "text"]].itertuples(index=False, name=None), True)
        participant_rows = df[df["speaker"] == "participant"]
        interviewer_rows = df[df["speaker"] == "interviewer"]

        participant_only = join_lines(
            participant_rows[["speaker", "text"]].itertuples(index=False, name=None),
            False,
        )
        interviewer_only = join_lines(
            interviewer_rows[["speaker", "text"]].itertuples(index=False, name=None),
            False,
        )
        interviewer_cleaned_rows = interviewer_rows[
            ~interviewer_rows["text"].str.contains(DIAGNOSTIC_RE, na=False)
        ]
        interviewer_cleaned = join_lines(
            interviewer_cleaned_rows[["speaker", "text"]].itertuples(index=False, name=None),
            False,
        )
        evidence_only = load_evidence_summary(evidence_json_dir, participant_id)

        for condition, text in [
            ("full_dialogue", full),
            ("participant_only", participant_only),
            ("interviewer_only", interviewer_only),
            ("interviewer_cleaned", interviewer_cleaned),
            ("evidence_only", evidence_only),
        ]:
            records.append(
                {
                    "participant_id": participant_id,
                    "input_condition": condition,
                    "text": text,
                }
            )

        stats.append(
            {
                "participant_id": participant_id,
                "turns_total": len(df),
                "participant_turns": len(participant_rows),
                "interviewer_turns": len(interviewer_rows),
                "diagnostic_interviewer_turns": int(
                    interviewer_rows["text"].str.contains(DIAGNOSTIC_RE, na=False).sum()
                ),
                "participant_words": int(len(participant_only.split())),
                "interviewer_words": int(len(interviewer_only.split())),
            }
        )

    return pd.DataFrame(records), pd.DataFrame(stats)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, scores)
    else:
        out["auc"] = np.nan
    return out


def get_scores(model: Pipeline, x: pd.Series) -> Optional[np.ndarray]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(clf, "decision_function"):
        return model.decision_function(x)
    return None


def make_models() -> Dict[str, Pipeline]:
    return {
        "tfidf_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def load_split(split_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if split_csv is None:
        return None
    split = pd.read_csv(split_csv)
    lower_map = {c.lower(): c for c in split.columns}
    id_col = lower_map.get("participant_id") or lower_map.get("participant") or lower_map.get("id")
    split_col = lower_map.get("split") or lower_map.get("set")
    if id_col is None or split_col is None:
        raise ValueError("split_csv must contain participant_id and split columns.")
    out = split[[id_col, split_col]].copy()
    out.columns = ["participant_id", "split"]
    out["participant_id"] = out["participant_id"].astype(str)
    out["split"] = out["split"].astype(str).str.lower()
    return out


def run_baselines(
    conditions: pd.DataFrame,
    labels: pd.DataFrame,
    split: Optional[pd.DataFrame],
    cv_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = conditions.merge(labels, on="participant_id", how="inner")
    data = data[data["text"].fillna("").str.strip() != ""].copy()
    if split is not None:
        data = data.merge(split, on="participant_id", how="left")

    metrics: List[Dict[str, object]] = []
    preds: List[Dict[str, object]] = []

    for condition, cdf in data.groupby("input_condition"):
        cdf = cdf.reset_index(drop=True)
        if cdf["label"].nunique() < 2:
            continue

        models = make_models()
        if split is not None and {"train", "dev"}.issubset(set(cdf["split"].dropna())):
            train_idx = cdf.index[cdf["split"] == "train"].to_numpy()
            test_idx = cdf.index[cdf["split"].isin(["dev", "test"])].to_numpy()
            fold_iter = [("heldout", train_idx, test_idx)]
        else:
            min_class_count = int(cdf["label"].value_counts().min())
            n_splits = min(cv_folds, min_class_count)
            if n_splits < 2:
                continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260611)
            fold_iter = [
                (f"fold_{i}", train_idx, test_idx)
                for i, (train_idx, test_idx) in enumerate(skf.split(cdf["text"], cdf["label"]), start=1)
            ]

        for model_name, model in models.items():
            for fold, train_idx, test_idx in fold_iter:
                train = cdf.iloc[train_idx]
                test = cdf.iloc[test_idx]
                model.fit(train["text"], train["label"])
                y_pred = model.predict(test["text"])
                scores = get_scores(model, test["text"])
                row = metric_dict(test["label"].to_numpy(), y_pred, scores)
                row.update(
                    {
                        "model": model_name,
                        "input_condition": condition,
                        "fold": fold,
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                )
                metrics.append(row)
                for participant_id, y_true, pred, score in zip(
                    test["participant_id"],
                    test["label"],
                    y_pred,
                    scores if scores is not None else [np.nan] * len(test),
                ):
                    preds.append(
                        {
                            "participant_id": participant_id,
                            "model": model_name,
                            "input_condition": condition,
                            "fold": fold,
                            "true_label": int(y_true),
                            "pred_label": int(pred),
                            "score": float(score) if not pd.isna(score) else np.nan,
                        }
                    )

    return pd.DataFrame(metrics), pd.DataFrame(preds)


def export_evidence_extraction_inputs(conditions: pd.DataFrame, output_dir: Path) -> None:
    participant = conditions[conditions["input_condition"] == "participant_only"]
    out_path = output_dir / "participant_for_evidence_extraction.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in participant.itertuples(index=False):
            f.write(
                json.dumps(
                    {
                        "participant_id": row.participant_id,
                        "input_condition": row.input_condition,
                        "text": row.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-dir", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--evidence-json-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = prepare_labels(args.labels_csv)
    split = load_split(args.split_csv)
    conditions, transcript_stats = build_conditions(args.transcripts_dir, args.evidence_json_dir)

    conditions.to_csv(args.output_dir / "conditions_long.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(args.output_dir / "labels_prepared.csv", index=False, encoding="utf-8-sig")
    transcript_stats.to_csv(args.output_dir / "transcript_stats.csv", index=False, encoding="utf-8-sig")
    export_evidence_extraction_inputs(conditions, args.output_dir)

    metrics, predictions = run_baselines(conditions, labels, split, args.cv_folds)
    metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote outputs to: {args.output_dir}")
    print("Next step: run symptom evidence extraction, then rerun with --evidence-json-dir.")


if __name__ == "__main__":
    main()
