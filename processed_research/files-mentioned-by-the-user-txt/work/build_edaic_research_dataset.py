from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


EDAIC_ROOT = Path(r"F:\数据库\E-daic")
LABELS_DIR = EDAIC_ROOT / "labels"
EDAIC_DATA_DIR = EDAIC_ROOT / "data" / "data"
DAIC_TRANSCRIPT_DIR = EDAIC_ROOT / "DAIC-WOZ_original_transcripts"
OUTPUT_DIR = EDAIC_ROOT / "processed_research"

EXPECTED_IDS = [i for i in range(300, 493) if i not in {342, 394, 398, 460}]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_speaker(speaker: str) -> str:
    lowered = normalize_space(speaker).lower()
    if lowered in {"ellie", "interviewer"}:
        return "interviewer"
    if lowered in {"participant", "subject", "user"}:
        return "participant"
    return "other"


def read_transcript(path: Path, participant_id: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)
        turns: list[dict[str, Any]] = []
        for idx, row in enumerate(reader, start=1):
            start_time = as_float(row.get("start_time") or row.get("Start_Time"))
            stop_time = as_float(row.get("stop_time") or row.get("End_Time"))
            raw_text = row.get("value") or row.get("Text") or ""
            speaker_raw = row.get("speaker") or row.get("Speaker") or ""
            text = normalize_space(raw_text)
            speaker_norm = normalize_speaker(speaker_raw)
            duration = None
            if start_time is not None and stop_time is not None:
                duration = max(0.0, stop_time - start_time)
            turns.append(
                {
                    "participant_id": participant_id,
                    "turn_index": idx,
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "duration_sec": duration,
                    "speaker_raw": normalize_space(speaker_raw),
                    "speaker_norm": speaker_norm,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                }
            )
    return turns


def load_label_maps() -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    detailed: dict[int, dict[str, str]] = {}
    for row in read_csv_dicts(LABELS_DIR / "detailed_lables.csv"):
        pid = as_int(row.get("Participant"))
        if pid in EXPECTED_IDS:
            detailed[pid] = row

    split_rows: dict[int, dict[str, str]] = {}
    for split_name, filename in [
        ("train", "train_split.csv"),
        ("dev", "dev_split.csv"),
        ("test", "test_split.csv"),
    ]:
        for row in read_csv_dicts(LABELS_DIR / filename):
            pid = as_int(row.get("Participant_ID"))
            if pid in EXPECTED_IDS:
                row = dict(row)
                row["split_file"] = split_name
                split_rows[pid] = row
    return detailed, split_rows


def load_metadata_map() -> dict[int, dict[str, str]]:
    path = LABELS_DIR / "metadata_mapped.csv"
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    for row in read_csv_dicts(path):
        pid = as_int(row.get("Participant_ID"))
        if pid in EXPECTED_IDS:
            out[pid] = row
    return out


def speaker_tagged(turns: list[dict[str, Any]]) -> str:
    pieces = []
    for turn in turns:
        speaker = turn["speaker_raw"] or turn["speaker_norm"]
        text = turn["text"]
        if text:
            pieces.append(f"[{speaker}] {text}")
    return " ".join(pieces)


def text_only(turns: list[dict[str, Any]], speaker_norm: str) -> str:
    return " ".join(turn["text"] for turn in turns if turn["speaker_norm"] == speaker_norm and turn["text"])


def build_qa_pairs(participant_id: int, turns: list[dict[str, Any]], label_base: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    i = 0
    pair_index = 1
    while i < len(turns):
        if turns[i]["speaker_norm"] != "interviewer":
            i += 1
            continue

        interviewer_block: list[dict[str, Any]] = []
        while i < len(turns) and turns[i]["speaker_norm"] == "interviewer":
            interviewer_block.append(turns[i])
            i += 1

        participant_block: list[dict[str, Any]] = []
        while i < len(turns) and turns[i]["speaker_norm"] == "participant":
            participant_block.append(turns[i])
            i += 1

        if not participant_block:
            continue

        def first_time(block: list[dict[str, Any]], key: str) -> Any:
            values = [turn[key] for turn in block if turn[key] is not None]
            return values[0] if values else ""

        def last_time(block: list[dict[str, Any]], key: str) -> Any:
            values = [turn[key] for turn in block if turn[key] is not None]
            return values[-1] if values else ""

        interviewer_text = " ".join(t["text"] for t in interviewer_block if t["text"])
        participant_text = " ".join(t["text"] for t in participant_block if t["text"])
        pairs.append(
            {
                "participant_id": participant_id,
                "split": label_base.get("split", ""),
                "phq_score": label_base.get("phq_score", ""),
                "phq_binary": label_base.get("phq_binary", ""),
                "qa_pair_index": pair_index,
                "interviewer_turn_count": len(interviewer_block),
                "participant_turn_count": len(participant_block),
                "interviewer_start_time": first_time(interviewer_block, "start_time"),
                "interviewer_stop_time": last_time(interviewer_block, "stop_time"),
                "participant_start_time": first_time(participant_block, "start_time"),
                "participant_stop_time": last_time(participant_block, "stop_time"),
                "interviewer_text": interviewer_text,
                "participant_text": participant_text,
                "interviewer_char_count": len(interviewer_text),
                "participant_char_count": len(participant_text),
            }
        )
        pair_index += 1
    return pairs


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("split", "")) for row in rows).items()))


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
    detailed, split_rows = load_label_maps()
    metadata = load_metadata_map()

    participant_rows: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []
    full_text_rows: list[dict[str, Any]] = []
    participant_text_rows: list[dict[str, Any]] = []
    interviewer_text_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []

    missing_labels: list[int] = []
    missing_transcripts: list[int] = []
    bad_transcripts: list[dict[str, Any]] = []
    raw_speaker_counter: Counter[str] = Counter()
    norm_speaker_counter: Counter[str] = Counter()

    for pid in EXPECTED_IDS:
        detailed_row = detailed.get(pid, {})
        split_row = split_rows.get(pid, {})
        meta_row = metadata.get(pid, {})
        if not detailed_row and not split_row:
            missing_labels.append(pid)

        split = detailed_row.get("split") or split_row.get("split_file") or ""
        phq_score = as_int(split_row.get("PHQ_Score"), as_int(detailed_row.get("Depression_severity")))
        phq_binary = as_int(split_row.get("PHQ_Binary"), as_int(detailed_row.get("Depression_label")))
        label_base = {
            "split": split,
            "phq_score": phq_score if phq_score is not None else "",
            "phq_binary": phq_binary if phq_binary is not None else "",
        }

        daic_transcript_path = DAIC_TRANSCRIPT_DIR / f"{pid}_TRANSCRIPT.csv"
        edaic_participant_dir = EDAIC_DATA_DIR / f"{pid}_P.tar" / f"{pid}_P"
        edaic_transcript_path = edaic_participant_dir / f"{pid}_Transcript.csv"
        audio_path = edaic_participant_dir / f"{pid}_AUDIO.wav"

        turns: list[dict[str, Any]] = []
        if daic_transcript_path.exists():
            try:
                turns = read_transcript(daic_transcript_path, pid)
            except Exception as exc:
                bad_transcripts.append({"participant_id": pid, "problem": repr(exc)})
        else:
            missing_transcripts.append(pid)

        for turn in turns:
            turn["split"] = split
            turn["phq_score"] = phq_score if phq_score is not None else ""
            turn["phq_binary"] = phq_binary if phq_binary is not None else ""
            raw_speaker_counter[turn["speaker_raw"]] += 1
            norm_speaker_counter[turn["speaker_norm"]] += 1
        all_turns.extend(turns)

        participant_turns = [t for t in turns if t["speaker_norm"] == "participant"]
        interviewer_turns = [t for t in turns if t["speaker_norm"] == "interviewer"]
        other_turns = [t for t in turns if t["speaker_norm"] == "other"]
        starts = [t["start_time"] for t in turns if t["start_time"] is not None]
        stops = [t["stop_time"] for t in turns if t["stop_time"] is not None]
        full_text = speaker_tagged(turns)
        participant_text = text_only(turns, "participant")
        interviewer_text = text_only(turns, "interviewer")

        common_text_fields = {
            "participant_id": pid,
            "split": split,
            "gender": split_row.get("Gender") or detailed_row.get("gender") or meta_row.get("Gender") or "",
            "age": detailed_row.get("age", ""),
            "phq_score": phq_score if phq_score is not None else "",
            "phq_binary": phq_binary if phq_binary is not None else "",
            "turn_count": len(turns),
            "participant_turn_count": len(participant_turns),
            "interviewer_turn_count": len(interviewer_turns),
            "other_turn_count": len(other_turns),
        }
        full_text_rows.append({**common_text_fields, "text": full_text, "char_count": len(full_text)})
        participant_text_rows.append({**common_text_fields, "text": participant_text, "char_count": len(participant_text)})
        interviewer_text_rows.append({**common_text_fields, "text": interviewer_text, "char_count": len(interviewer_text)})
        qa_rows.extend(build_qa_pairs(pid, turns, label_base))

        participant_rows.append(
            {
                "participant_id": pid,
                "split": split,
                "avec_participant_id": meta_row.get("AVECParticipant_ID", ""),
                "gender": split_row.get("Gender") or detailed_row.get("gender") or meta_row.get("Gender") or "",
                "age": detailed_row.get("age", ""),
                "phq_score": phq_score if phq_score is not None else "",
                "phq_binary": phq_binary if phq_binary is not None else "",
                "ptsd_score": split_row.get("PTSD Severity") or detailed_row.get("PTSD_severity") or meta_row.get("PTSD Severity") or "",
                "ptsd_binary": split_row.get("PCL-C (PTSD)") or detailed_row.get("PTSD_label") or meta_row.get("PCL-C (PTSD)") or "",
                "phq8_no_interest": detailed_row.get("PHQ8_1_NoInterest", ""),
                "phq8_depressed": detailed_row.get("PHQ8_2_Depressed", ""),
                "phq8_sleep": detailed_row.get("PHQ8_3_Sleep", ""),
                "phq8_tired": detailed_row.get("PHQ8_4_Tired", ""),
                "phq8_appetite": detailed_row.get("PHQ8_5_Appetite", ""),
                "phq8_failure": detailed_row.get("PHQ8_6_Failure", ""),
                "phq8_concentration": detailed_row.get("PHQ8_7_Concentration", ""),
                "phq8_psychomotor": detailed_row.get("PHQ8_8_Psychomotor", ""),
                "edaic_participant_dir": str(edaic_participant_dir),
                "edaic_transcript_path": str(edaic_transcript_path),
                "daic_speaker_transcript_path": str(daic_transcript_path),
                "audio_path": str(audio_path),
                "has_edaic_transcript": edaic_transcript_path.exists(),
                "has_daic_speaker_transcript": daic_transcript_path.exists(),
                "has_audio": audio_path.exists(),
                "turn_count": len(turns),
                "participant_turn_count": len(participant_turns),
                "interviewer_turn_count": len(interviewer_turns),
                "other_turn_count": len(other_turns),
                "qa_pair_count": sum(1 for row in qa_rows if row["participant_id"] == pid),
                "full_text_char_count": len(full_text),
                "participant_text_char_count": len(participant_text),
                "interviewer_text_char_count": len(interviewer_text),
                "first_start_time": min(starts) if starts else "",
                "last_stop_time": max(stops) if stops else "",
            }
        )
        if not edaic_transcript_path.exists():
            quality_flags.append(
                {"participant_id": pid, "flag": "missing_edaic_transcript", "detail": str(edaic_transcript_path)}
            )
        if not audio_path.exists():
            quality_flags.append({"participant_id": pid, "flag": "missing_audio", "detail": str(audio_path)})
        if not turns:
            quality_flags.append({"participant_id": pid, "flag": "empty_or_unreadable_daic_transcript", "detail": ""})
        if turns and not interviewer_turns:
            quality_flags.append(
                {
                    "participant_id": pid,
                    "flag": "no_interviewer_turns",
                    "detail": "Original DAIC-WOZ speaker transcript contains participant turns only.",
                }
            )
        if turns and not participant_turns:
            quality_flags.append(
                {
                    "participant_id": pid,
                    "flag": "no_participant_turns",
                    "detail": "Original DAIC-WOZ speaker transcript contains interviewer turns only.",
                }
            )
        if turns and not any(row["participant_id"] == pid for row in qa_rows):
            quality_flags.append(
                {
                    "participant_id": pid,
                    "flag": "no_question_answer_pairs",
                    "detail": "No interviewer block followed by participant block was found.",
                }
            )

    participant_fields = [
        "participant_id",
        "split",
        "avec_participant_id",
        "gender",
        "age",
        "phq_score",
        "phq_binary",
        "ptsd_score",
        "ptsd_binary",
        "phq8_no_interest",
        "phq8_depressed",
        "phq8_sleep",
        "phq8_tired",
        "phq8_appetite",
        "phq8_failure",
        "phq8_concentration",
        "phq8_psychomotor",
        "edaic_participant_dir",
        "edaic_transcript_path",
        "daic_speaker_transcript_path",
        "audio_path",
        "has_edaic_transcript",
        "has_daic_speaker_transcript",
        "has_audio",
        "turn_count",
        "participant_turn_count",
        "interviewer_turn_count",
        "other_turn_count",
        "qa_pair_count",
        "full_text_char_count",
        "participant_text_char_count",
        "interviewer_text_char_count",
        "first_start_time",
        "last_stop_time",
    ]
    turn_fields = [
        "participant_id",
        "split",
        "phq_score",
        "phq_binary",
        "turn_index",
        "start_time",
        "stop_time",
        "duration_sec",
        "speaker_raw",
        "speaker_norm",
        "text",
        "char_count",
        "word_count",
    ]
    text_fields = [
        "participant_id",
        "split",
        "gender",
        "age",
        "phq_score",
        "phq_binary",
        "turn_count",
        "participant_turn_count",
        "interviewer_turn_count",
        "other_turn_count",
        "char_count",
        "text",
    ]
    qa_fields = [
        "participant_id",
        "split",
        "phq_score",
        "phq_binary",
        "qa_pair_index",
        "interviewer_turn_count",
        "participant_turn_count",
        "interviewer_start_time",
        "interviewer_stop_time",
        "participant_start_time",
        "participant_stop_time",
        "interviewer_text",
        "participant_text",
        "interviewer_char_count",
        "participant_char_count",
    ]

    write_csv_dicts(OUTPUT_DIR / "participant_index.csv", participant_rows, participant_fields)
    write_csv_dicts(OUTPUT_DIR / "turns.csv", all_turns, turn_fields)
    write_csv_dicts(OUTPUT_DIR / "texts_full_dialogue.csv", full_text_rows, text_fields)
    write_csv_dicts(OUTPUT_DIR / "texts_participant_only.csv", participant_text_rows, text_fields)
    write_csv_dicts(OUTPUT_DIR / "texts_interviewer_only.csv", interviewer_text_rows, text_fields)
    write_csv_dicts(OUTPUT_DIR / "question_answer_pairs.csv", qa_rows, qa_fields)
    write_csv_dicts(OUTPUT_DIR / "quality_flags.csv", quality_flags, ["participant_id", "flag", "detail"])

    phq_counter = Counter(str(row.get("phq_binary", "")) for row in participant_rows)
    participant_text_lengths = [int(row["participant_text_char_count"]) for row in participant_rows]
    interviewer_text_lengths = [int(row["interviewer_text_char_count"]) for row in participant_rows]
    qa_counts = [int(row["qa_pair_count"]) for row in participant_rows]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUTPUT_DIR),
        "source_dirs": {
            "edaic_root": str(EDAIC_ROOT),
            "labels_dir": str(LABELS_DIR),
            "daic_speaker_transcripts": str(DAIC_TRANSCRIPT_DIR),
        },
        "participants": {
            "expected_daic_woz_ids": len(EXPECTED_IDS),
            "participant_index_rows": len(participant_rows),
            "split_counts": split_counts(participant_rows),
            "phq_binary_counts": dict(sorted(phq_counter.items())),
            "missing_label_count": len(missing_labels),
            "missing_label_ids": missing_labels,
        },
        "transcripts": {
            "missing_transcript_count": len(missing_transcripts),
            "missing_transcript_ids": missing_transcripts,
            "bad_transcript_count": len(bad_transcripts),
            "bad_transcripts": bad_transcripts,
            "turn_rows": len(all_turns),
            "speaker_raw_counts": dict(sorted(raw_speaker_counter.items())),
            "speaker_norm_counts": dict(sorted(norm_speaker_counter.items())),
        },
        "quality_flags": {
            "flag_rows": len(quality_flags),
            "flag_counts": dict(sorted(Counter(row["flag"] for row in quality_flags).items())),
            "flagged_participants": sorted({int(row["participant_id"]) for row in quality_flags}),
        },
        "derived_texts": {
            "full_dialogue_rows": len(full_text_rows),
            "participant_only_rows": len(participant_text_rows),
            "interviewer_only_rows": len(interviewer_text_rows),
            "question_answer_pair_rows": len(qa_rows),
            "participant_text_char_count": numeric_summary(participant_text_lengths),
            "interviewer_text_char_count": numeric_summary(interviewer_text_lengths),
            "qa_pair_count_per_participant": numeric_summary(qa_counts),
        },
        "important_notes": [
            "DAIC-WOZ 300-492 labels are train/dev only in these eDAIC label files; eDAIC test_split.csv contains 600+ IDs, so it is not mixed into this DAIC-WOZ processed set.",
            "Original DAIC-WOZ speaker transcripts are stored separately because Windows treats *_Transcript.csv and *_TRANSCRIPT.csv as the same filename even though their contents differ.",
            "Raw source data was not modified; this directory contains processed analysis copies.",
        ],
    }
    (OUTPUT_DIR / "processing_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# DAIC-WOZ / eDAIC 研究数据预处理报告",
        "",
        f"- 生成时间: {report['created_at']}",
        f"- 输出目录: `{OUTPUT_DIR}`",
        f"- participant 索引行数: {len(participant_rows)}",
        f"- split 分布: {report['participants']['split_counts']}",
        f"- PHQ 二分类分布: {report['participants']['phq_binary_counts']}",
        f"- turn-level 行数: {len(all_turns)}",
        f"- speaker 归一化计数: {report['transcripts']['speaker_norm_counts']}",
        f"- QA pair 行数: {len(qa_rows)}",
        "",
        "## 输出文件",
        "",
        "- `participant_index.csv`: 每个 participant 一行，合并 split、PHQ、PTSD、路径、turn 数和文本长度。",
        "- `turns.csv`: 原始 DAIC-WOZ speaker transcript 解析后的逐轮对话表。",
        "- `texts_full_dialogue.csv`: 带 speaker 标签的完整对话文本。",
        "- `texts_participant_only.csv`: 只保留 Participant 话语的文本。",
        "- `texts_interviewer_only.csv`: 只保留 Ellie/interviewer 话语的文本。",
        "- `question_answer_pairs.csv`: 连续 Ellie 话语块 + 紧随其后的 Participant 回答块。",
        "- `quality_flags.csv`: 逐 participant 的质量提示，例如没有 interviewer turn 的样本。",
        "- `processing_manifest.json`: 机器可读的处理摘要和校验结果。",
        "",
        "## 完整性检查",
        "",
        f"- 预期 DAIC-WOZ ID: {len(EXPECTED_IDS)}",
        f"- 缺标签: {len(missing_labels)}",
        f"- 缺 transcript: {len(missing_transcripts)}",
        f"- transcript 读取异常: {len(bad_transcripts)}",
        f"- 质量提示行数: {len(quality_flags)}",
        f"- 质量提示分布: {report['quality_flags']['flag_counts']}",
        "",
        "## 关键注意",
        "",
        "- 这 189 个 DAIC-WOZ 样本在当前标签文件中只有 `train` 和 `dev`，没有 DAIC-WOZ test 标签。",
        "- eDAIC 的 `test_split.csv` 是 600+ ID，不应混进 DAIC-WOZ 原始 189 人实验。",
        "- 原始 DAIC-WOZ 的 `*_TRANSCRIPT.csv` 被单独放在 `DAIC-WOZ_original_transcripts`，避免和 eDAIC 的 `*_Transcript.csv` 在 Windows 下发生大小写同名冲突。",
    ]
    (OUTPUT_DIR / "data_quality_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
