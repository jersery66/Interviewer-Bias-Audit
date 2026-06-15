from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


LABELS_DIR = Path(r"F:\数据库\E-daic\labels")
TRANSCRIPTS_DIR = Path(r"F:\数据库\E-daic\DAIC-WOZ_original_transcripts")
OUTPUT_DIR = Path(r"F:\数据库\DAIC-WOZ\processed_research")

EXPECTED_IDS = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]

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


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_speaker(raw: str) -> str:
    value = clean_text(raw).lower()
    if "ellie" in value or "interviewer" in value:
        return "interviewer"
    if "participant" in value:
        return "participant"
    return "other"


def load_labels() -> dict[int, dict[str, Any]]:
    detailed: dict[int, dict[str, str]] = {}
    detailed_path = LABELS_DIR / "detailed_lables.csv"
    for row in read_csv_dicts(detailed_path):
        pid = as_int(row.get("Participant"))
        if pid in EXPECTED_IDS:
            detailed[pid] = row

    split_map: dict[int, dict[str, str]] = {}
    for split_name, filename in [
        ("train", "train_split.csv"),
        ("dev", "dev_split.csv"),
        ("test", "test_split.csv"),
    ]:
        for row in read_csv_dicts(LABELS_DIR / filename):
            pid = as_int(row.get("Participant_ID"))
            if pid in EXPECTED_IDS:
                row = dict(row)
                row["split"] = split_name
                split_map[pid] = row

    labels: dict[int, dict[str, Any]] = {}
    for pid in EXPECTED_IDS:
        drow = detailed.get(pid, {})
        srow = split_map.get(pid, {})
        phq_score = as_int(srow.get("PHQ_Score"), as_int(drow.get("Depression_severity")))
        phq_binary = as_int(srow.get("PHQ_Binary"), as_int(drow.get("Depression_label")))
        labels[pid] = {
            "participant_id": pid,
            "split": drow.get("split") or srow.get("split") or "",
            "label": phq_binary if phq_binary is not None else "",
            "phq_binary": phq_binary if phq_binary is not None else "",
            "phq_score": phq_score if phq_score is not None else "",
            "gender": srow.get("Gender") or drow.get("gender") or "",
            "age": drow.get("age", ""),
            "ptsd_binary": srow.get("PCL-C (PTSD)") or drow.get("PTSD_label") or "",
            "ptsd_score": srow.get("PTSD Severity") or drow.get("PTSD_severity") or "",
        }
    return labels


def read_transcript(path: Path, participant_id: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(reader, start=1):
            start = as_float(row.get("start_time") or row.get("Start_Time"))
            stop = as_float(row.get("stop_time") or row.get("End_Time"))
            text = clean_text(row.get("value") or row.get("Text") or "")
            speaker_raw = clean_text(row.get("speaker") or row.get("Speaker") or "")
            speaker_norm = normalize_speaker(speaker_raw)
            rows.append(
                {
                    "participant_id": participant_id,
                    "turn_index": idx,
                    "start_time": start if start is not None else "",
                    "stop_time": stop if stop is not None else "",
                    "duration_sec": round(max(0.0, stop - start), 3) if start is not None and stop is not None else "",
                    "speaker_raw": speaker_raw,
                    "speaker_norm": speaker_norm,
                    "text": text,
                    "is_diagnostic_interviewer_turn": int(
                        speaker_norm == "interviewer" and bool(DIAGNOSTIC_RE.search(text))
                    ),
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                }
            )
    return rows


def join_turns(turns: list[dict[str, Any]], include_speaker: bool) -> str:
    lines = []
    for turn in turns:
        text = turn["text"]
        if not text:
            continue
        if include_speaker:
            lines.append(f"{turn['speaker_norm']}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    participant_index: list[dict[str, Any]] = []
    labels_prepared: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []
    transcript_stats: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    evidence_inputs: list[dict[str, Any]] = []
    qa_pairs: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []

    for pid in EXPECTED_IDS:
        label = labels[pid]
        transcript_path = TRANSCRIPTS_DIR / f"{pid}_TRANSCRIPT.csv"
        if not transcript_path.exists():
            quality_flags.append({"participant_id": pid, "flag": "missing_transcript", "detail": str(transcript_path)})
            turns: list[dict[str, Any]] = []
        else:
            turns = read_transcript(transcript_path, pid)

        for turn in turns:
            turn["split"] = label["split"]
            turn["label"] = label["label"]
            turn["phq_score"] = label["phq_score"]
        all_turns.extend(turns)

        participant_turns = [t for t in turns if t["speaker_norm"] == "participant"]
        interviewer_turns = [t for t in turns if t["speaker_norm"] == "interviewer"]
        diagnostic_interviewer_turns = [t for t in interviewer_turns if t["is_diagnostic_interviewer_turn"]]
        interviewer_cleaned_turns = [t for t in interviewer_turns if not t["is_diagnostic_interviewer_turn"]]

        full_dialogue = join_turns(turns, include_speaker=True)
        participant_only = join_turns(participant_turns, include_speaker=False)
        interviewer_only = join_turns(interviewer_turns, include_speaker=False)
        interviewer_cleaned = join_turns(interviewer_cleaned_turns, include_speaker=False)

        pair_index = 1
        cursor = 0
        while cursor < len(turns):
            if turns[cursor]["speaker_norm"] != "interviewer":
                cursor += 1
                continue
            interviewer_block: list[dict[str, Any]] = []
            while cursor < len(turns) and turns[cursor]["speaker_norm"] == "interviewer":
                interviewer_block.append(turns[cursor])
                cursor += 1
            participant_block: list[dict[str, Any]] = []
            while cursor < len(turns) and turns[cursor]["speaker_norm"] == "participant":
                participant_block.append(turns[cursor])
                cursor += 1
            if not participant_block:
                continue
            interviewer_text = " ".join(t["text"] for t in interviewer_block if t["text"])
            participant_text = " ".join(t["text"] for t in participant_block if t["text"])
            qa_pairs.append(
                {
                    "participant_id": pid,
                    "split": label["split"],
                    "label": label["label"],
                    "phq_score": label["phq_score"],
                    "qa_pair_index": pair_index,
                    "interviewer_turn_count": len(interviewer_block),
                    "participant_turn_count": len(participant_block),
                    "interviewer_text": interviewer_text,
                    "participant_text": participant_text,
                    "interviewer_char_count": len(interviewer_text),
                    "participant_char_count": len(participant_text),
                }
            )
            pair_index += 1

        for condition, text in [
            ("full_dialogue", full_dialogue),
            ("participant_only", participant_only),
            ("interviewer_only", interviewer_only),
            ("interviewer_cleaned", interviewer_cleaned),
        ]:
            conditions.append(
                {
                    "participant_id": pid,
                    "split": label["split"],
                    "label": label["label"],
                    "phq_score": label["phq_score"],
                    "input_condition": condition,
                    "condition_status": "ready",
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                }
            )

        evidence_inputs.append(
            {
                "task_id": f"{pid}_symptom_evidence_extraction",
                "participant_id": pid,
                "input_condition": "participant_only",
                "prompt_version": "symptom_evidence_v1",
                "text": participant_only,
            }
        )

        if turns and not interviewer_turns:
            quality_flags.append(
                {
                    "participant_id": pid,
                    "flag": "no_interviewer_turns",
                    "detail": "Original DAIC-WOZ transcript has participant turns only; exclude from interviewer-only analyses.",
                }
            )
        if turns and not participant_turns:
            quality_flags.append(
                {
                    "participant_id": pid,
                    "flag": "no_participant_turns",
                    "detail": "Original DAIC-WOZ transcript has interviewer turns only.",
                }
            )

        participant_words = sum(t["word_count"] for t in participant_turns)
        interviewer_words = sum(t["word_count"] for t in interviewer_turns)
        interviewer_cleaned_words = sum(t["word_count"] for t in interviewer_cleaned_turns)
        transcript_stats.append(
            {
                "participant_id": pid,
                "split": label["split"],
                "label": label["label"],
                "phq_score": label["phq_score"],
                "turns_total": len(turns),
                "participant_turns": len(participant_turns),
                "interviewer_turns": len(interviewer_turns),
                "diagnostic_interviewer_turns": len(diagnostic_interviewer_turns),
                "interviewer_cleaned_turns": len(interviewer_cleaned_turns),
                "participant_words": participant_words,
                "interviewer_words": interviewer_words,
                "interviewer_cleaned_words": interviewer_cleaned_words,
                "transcript_path": str(transcript_path),
            }
        )
        participant_index.append(
            {
                "participant_id": pid,
                "split": label["split"],
                "label": label["label"],
                "phq_binary": label["phq_binary"],
                "phq_score": label["phq_score"],
                "gender": label["gender"],
                "age": label["age"],
                "ptsd_binary": label["ptsd_binary"],
                "ptsd_score": label["ptsd_score"],
                "transcript_path": str(transcript_path),
                "has_transcript": transcript_path.exists(),
                "turns_total": len(turns),
                "participant_turns": len(participant_turns),
                "interviewer_turns": len(interviewer_turns),
                "diagnostic_interviewer_turns": len(diagnostic_interviewer_turns),
                "interviewer_cleaned_turns": len(interviewer_cleaned_turns),
                "participant_words": participant_words,
                "interviewer_words": interviewer_words,
                "interviewer_cleaned_words": interviewer_cleaned_words,
            }
        )
        labels_prepared.append(
            {
                "participant_id": pid,
                "label": label["label"],
                "phq_binary": label["phq_binary"],
                "phq_score": label["phq_score"],
            }
        )
        split_rows.append({"participant_id": pid, "split": label["split"]})

    condition_status_rows = [
        {
            "input_condition": "full_dialogue",
            "status": "ready",
            "source": "DAIC-WOZ original speaker transcript",
            "note": "C1 in original plan.",
        },
        {
            "input_condition": "participant_only",
            "status": "ready",
            "source": "DAIC-WOZ original speaker transcript",
            "note": "C2 in original plan.",
        },
        {
            "input_condition": "interviewer_only",
            "status": "ready",
            "source": "DAIC-WOZ original speaker transcript",
            "note": "C3 in original plan; participants flagged in quality_flags should be excluded for this condition.",
        },
        {
            "input_condition": "interviewer_cleaned",
            "status": "ready_rule_based",
            "source": "Keyword removal from interviewer_only",
            "note": "C4 in original plan; operational rule matches outputs/新研究_实验代码骨架.py.",
        },
        {
            "input_condition": "symptom_evidence_only",
            "status": "pending_llm_extraction",
            "source": "participant_for_evidence_extraction.jsonl",
            "note": "C5 in original plan; do not run as baseline until LLM evidence JSON has been generated and audited.",
        },
    ]

    participant_fields = [
        "participant_id",
        "split",
        "label",
        "phq_binary",
        "phq_score",
        "gender",
        "age",
        "ptsd_binary",
        "ptsd_score",
        "transcript_path",
        "has_transcript",
        "turns_total",
        "participant_turns",
        "interviewer_turns",
        "diagnostic_interviewer_turns",
        "interviewer_cleaned_turns",
        "participant_words",
        "interviewer_words",
        "interviewer_cleaned_words",
    ]
    turn_fields = [
        "participant_id",
        "split",
        "label",
        "phq_score",
        "turn_index",
        "start_time",
        "stop_time",
        "duration_sec",
        "speaker_raw",
        "speaker_norm",
        "is_diagnostic_interviewer_turn",
        "text",
        "char_count",
        "word_count",
    ]
    condition_fields = [
        "participant_id",
        "split",
        "label",
        "phq_score",
        "input_condition",
        "condition_status",
        "text",
        "char_count",
        "word_count",
    ]
    stats_fields = [
        "participant_id",
        "split",
        "label",
        "phq_score",
        "turns_total",
        "participant_turns",
        "interviewer_turns",
        "diagnostic_interviewer_turns",
        "interviewer_cleaned_turns",
        "participant_words",
        "interviewer_words",
        "interviewer_cleaned_words",
        "transcript_path",
    ]

    write_csv(OUTPUT_DIR / "participant_index.csv", participant_index, participant_fields)
    write_csv(OUTPUT_DIR / "labels_prepared.csv", labels_prepared, ["participant_id", "label", "phq_binary", "phq_score"])
    write_csv(OUTPUT_DIR / "split.csv", split_rows, ["participant_id", "split"])
    write_csv(OUTPUT_DIR / "turns.csv", all_turns, turn_fields)
    write_csv(OUTPUT_DIR / "conditions_long.csv", conditions, condition_fields)
    write_csv(OUTPUT_DIR / "transcript_stats.csv", transcript_stats, stats_fields)
    write_csv(OUTPUT_DIR / "condition_status.csv", condition_status_rows, ["input_condition", "status", "source", "note"])
    write_csv(OUTPUT_DIR / "quality_flags.csv", quality_flags, ["participant_id", "flag", "detail"])
    write_csv(
        OUTPUT_DIR / "question_answer_pairs.csv",
        qa_pairs,
        [
            "participant_id",
            "split",
            "label",
            "phq_score",
            "qa_pair_index",
            "interviewer_turn_count",
            "participant_turn_count",
            "interviewer_text",
            "participant_text",
            "interviewer_char_count",
            "participant_char_count",
        ],
    )
    write_jsonl(OUTPUT_DIR / "participant_for_evidence_extraction.jsonl", evidence_inputs)

    split_counts = Counter(row["split"] for row in participant_index)
    label_counts = Counter(str(row["label"]) for row in participant_index)
    condition_counts = Counter(row["input_condition"] for row in conditions)
    condition_nonempty_counts = Counter(row["input_condition"] for row in conditions if clean_text(str(row["text"])))
    flag_counts = Counter(row["flag"] for row in quality_flags)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_focus": "DAIC-WOZ original speaker-separated transcripts",
        "output_dir": str(OUTPUT_DIR),
        "source_transcripts_dir": str(TRANSCRIPTS_DIR),
        "source_labels_dir": str(LABELS_DIR),
        "participants": {
            "expected_count": len(EXPECTED_IDS),
            "index_rows": len(participant_index),
            "split_counts": dict(sorted(split_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "turns": {
            "rows": len(all_turns),
            "speaker_counts": dict(sorted(Counter(t["speaker_norm"] for t in all_turns).items())),
        },
        "conditions": {
            "ready_rows": len(conditions),
            "condition_counts": dict(sorted(condition_counts.items())),
            "condition_nonempty_counts": dict(sorted(condition_nonempty_counts.items())),
            "symptom_evidence_only": "pending_llm_extraction",
            "question_answer_pair_rows": len(qa_pairs),
        },
        "quality_flags": {
            "rows": len(quality_flags),
            "flag_counts": dict(sorted(flag_counts.items())),
            "flagged_participants": sorted({int(row["participant_id"]) for row in quality_flags}),
        },
        "text_length_summaries": {
            "participant_words": numeric_summary([int(row["participant_words"]) for row in transcript_stats]),
            "interviewer_words": numeric_summary([int(row["interviewer_words"]) for row in transcript_stats]),
            "interviewer_cleaned_words": numeric_summary([int(row["interviewer_cleaned_words"]) for row in transcript_stats]),
        },
        "interpretation": [
            "This is the DAIC-WOZ research dataset requested by the original plan.",
            "eDAIC is used only as the local source of labels and storage context; eDAIC ASR transcripts are not used to build conditions.",
            "C1-C4 are ready. C5 symptom_evidence_only requires LLM evidence extraction from participant_for_evidence_extraction.jsonl.",
            "DAIC-WOZ IDs 451, 458, and 480 have no interviewer turns in their original transcript and should be excluded from interviewer-only analyses.",
        ],
    }
    (OUTPUT_DIR / "processing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# DAIC-WOZ 研究数据核验报告",
        "",
        f"- 生成时间: {manifest['created_at']}",
        "- 研究主数据集: DAIC-WOZ original speaker-separated transcripts",
        f"- 输出目录: `{OUTPUT_DIR}`",
        f"- 样本数: {len(participant_index)}",
        f"- split 分布: {dict(sorted(split_counts.items()))}",
        f"- PHQ 二分类分布: {dict(sorted(label_counts.items()))}",
        f"- turn 行数: {len(all_turns)}",
        f"- speaker 分布: {manifest['turns']['speaker_counts']}",
        "",
        "## 与最初研究方案的对应关系",
        "",
        "- C1 `full_dialogue`: 已生成，见 `conditions_long.csv`。",
        "- C2 `participant_only`: 已生成，见 `conditions_long.csv`。",
        "- C3 `interviewer_only`: 已生成，见 `conditions_long.csv`；451/458/480 没有 interviewer turn，需排除或单独报告。",
        "- C4 `interviewer_cleaned`: 已按最初代码骨架的关键词规则生成，见 `conditions_long.csv` 和 `transcript_stats.csv`。",
        "- C5 `symptom_evidence_only`: 还不能直接生成，因为需要先跑 LLM 症状证据抽取；输入文件已准备为 `participant_for_evidence_extraction.jsonl`。",
        "",
        "## 当前文件是否研究需要",
        "",
        "- `participant_index.csv`: 需要。样本、标签、split 与 transcript 统计总表。",
        "- `labels_prepared.csv`: 需要。传统模型和 LLM 评估用标签。",
        "- `split.csv`: 需要。固定 train/dev 划分。",
        "- `turns.csv`: 需要。所有输入条件和后续证据定位的基础。",
        "- `conditions_long.csv`: 需要。C1-C4 的标准长表，下一步 baseline 直接读这个。",
        "- `transcript_stats.csv`: 需要。论文表 1/表 2 和诊断性问题数量统计。",
        "- `participant_for_evidence_extraction.jsonl`: 需要。生成 C5 symptom evidence-only 的 LLM 输入。",
        "- `quality_flags.csv`: 需要。记录不能进入某些条件分析的样本。",
        "- `question_answer_pairs.csv`: 辅助需要。用于访谈者问题-患者回答配对分析和案例解释。",
        "- `condition_status.csv`: 辅助需要。说明哪些条件 ready，哪些 pending。",
        "",
        "## 不应作为当前主研究输入的内容",
        "",
        "- eDAIC 的 `*_Transcript.csv` 不用于本研究主条件构造，因为它没有 speaker。",
        "- eDAIC/DAIC-WOZ 的音频和大体积 multimodal features 暂不需要，除非以后扩展到声学/视觉模型。",
        "- `symptom_evidence_only` 不应被假造；必须等 LLM 抽取后再加入 baseline。",
    ]
    (OUTPUT_DIR / "research_need_verification.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
