"""Prepare packet_3 for the staged human validation of the D-P proxy.

Packet 3 preserves the original packet-2 main sample, adds a separately
analysed rare-domain enrichment sample, and includes participant 385 as an
owner-side sentinel.  All three sample types receive opaque IDs in one blinded
stage-1 packet.  No C5 span is exposed before human consensus is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reanalysis_v2.c5_blind_workflow import build_blind_cases, build_line_index
from reanalysis_v2.c5_blind_workflow_v3 import (
    DOMAIN_FEATURES,
    RARE_TARGETS,
    build_stage1_review_form,
    build_validation_sample,
    make_opaque_blind_key,
    select_enrichment_sample,
    validate_stage1_reviewer_form,
)


ANALYSIS_ROOT = REPO_ROOT / "analysis_v2"
AUDIT_ROOT = ANALYSIS_ROOT / "04_c5_controls" / "blind_quality_audit"
DEFAULT_PUBLIC_PACKET = AUDIT_ROOT / "packet_3"
DEFAULT_PRIVATE_PACKET = AUDIT_ROOT / "private" / "packet_3"
PACKET2_PRIVATE = AUDIT_ROOT / "private" / "packet_2"
PARTICIPANTS_PATH = ANALYSIS_ROOT / "01_inputs" / "c2_participant_speech.csv"
DOMAIN_COUNT_PATH = ANALYSIS_ROOT / "04_c5_controls" / "domain_count" / "input.csv"

BLIND_ID_SEED = 20260721
ENRICHMENT_TARGET_QUOTA = 4
ENRICHMENT_MIN_N = 10
ENRICHMENT_MAX_N = 15
SENTINEL_PARTICIPANT_ID = 385
REVIEWER_A_ORDER_SEED = 202607211
REVIEWER_B_ORDER_SEED = 202607212


def _main_repo_root() -> Path:
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="strict",
    ).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = REPO_ROOT / common_path
    return common_path.resolve().parent


DEFAULT_REVIEWED_SPANS_PATH = (
    _main_repo_root()
    / "processed_research"
    / "verified_source_data_official142_20260703"
    / "03_canonical_text_official142"
    / "reviewed_c5_spans_official142.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="strict",
        ).strip()
    except Exception:
        return "unknown"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _codebook_text() -> str:
    return """# D-P人工效度验证第一阶段码本（packet 3）

## 第一阶段目标

只根据完整患者语言建立人工参照。评价者不得查看C5证据、D-P presence/count、
PHQ-8标签、预测结果或样本类型。每条证据必须复制患者原话并填写1-based行号。
多条独立证据使用 ` || ` 分隔，行号按同样顺序分隔。

## 通用症状域状态

适用于兴趣减退、食欲/体重、注意力/精神运动、抑郁情绪、功能损害、
自我价值/内疚、睡眠/疲劳/精力和自杀/自伤：

- `current_present`：当前症状有明确患者证据；
- `explicitly_absent`：患者明确否认当前症状；
- `past_only`：只有既往表现，没有当前阳性证据；
- `conflicting`：当前证据前后冲突，无法形成单一判断；
- `indeterminate`：无法判断。

## 域特异状态

`mental_health_history`仅使用：

- `history_present`：本人既往心理健康诊断、治疗、咨询、住院或相关病史存在；
- `explicitly_no_history`：本人明确表示无相关既往史；
- `indeterminate`：无法判断。

`protective_or_absent_symptom`只判断保护性或否认性证据是否出现：

- `protective_denial_evidence_present`；
- `no_protective_denial_evidence`；
- `indeterminate`。

该域不表示抑郁症状阳性。

## indeterminate原因

只能选择：`not_mentioned`、`not_covered`、`answer_too_short`、
`insufficient_context`、`ambiguous_expression`、`other`。

## 计数和属性

- `count_bin`：独立证据事件数，使用 `0`、`1`、`2`、`3+`；即使为`3+`，
  仍应记录所有找到的实际证据，而不是只写三条。
- `temporality`：`current`、`past`、`mixed`、`unclear`、`not_applicable`。
- `negation`：`affirmed`、`explicitly_negated`、`mixed`、`unclear`、
  `not_applicable`。
- `subject`：`participant`、`other_person`、`mixed`、`unclear`、
  `not_applicable`。

## 冻结规则

两名评价者先独立完成培训并冻结码本，再独立完成正式第一阶段。原始表不得覆盖。
先计算一致性、形成并冻结人工共识；只有共识文件哈希冻结后，才允许生成第二阶段
C5证据审核表。
"""


def _protocol_text(main_n: int, enriched_n: int, validation_n: int) -> str:
    return f"""# D-P人工效度验证协议（packet 3）

## 样本结构

- 主验证样本：保留packet 2原有{main_n}人，主要总体效度指标只由该样本估计；
- 稀有域富集样本：{enriched_n}人，按最小覆盖原则补充预先指定的稀有D-P单元，
  单独报告，不与主样本直接合并计算总体准确率；
- 哨兵个案：ID 385在owner key中单独标记，不进入总体效度统计；
- 第一阶段共{validation_n}名验证参与者，每位评价者完成{validation_n * 10}个
  参与者×域单元。

评价者只看到统一的opaque blind ID，不看到主样本、富集样本或哨兵身份。

## 阶段一：人工参照

评价者只获得完整患者语言、逐行文本、域名称、冻结码本和空白表。严禁提供：
C5证据、D-P presence/count、参与者ID、PHQ-8标签或分数、模型结果、访谈者语言、
样本类型。两名评价者独立完成后，先计算一致性，再在仍未打开D-P的条件下形成共识。

## 共识冻结闸门

共识表必须保留A/B原始判断、共识状态、证据、计数、时间、否定、主体和裁决理由。
共识文件通过校验并写入SHA-256冻结记录后，第二阶段脚本才可读取C5 spans。

## 阶段二：C5证据与遗漏

脚本按抽样参与者的实际reviewed span行展开全部C5证据；不得用D-P count或`3+`
推算证据条数。逐条审核该片段是否有患者来源支持，以及说话者、目标域、否定、
时间、主体、上下文、重复和人工证据匹配。同时把阶段一共识中的人工证据事件展开为
遗漏审核表，判断重要证据是否被C5覆盖。

## 报告边界

主要presence效度、参与者聚类bootstrap区间和总体Cohen's kappa只报告主30人。
富集样本按稀有域单独描述；ID 385仅作哨兵个案。count结果为次要结果，报告四级
一致率、加权kappa、平均绝对差和Spearman相关，不作为临床效度的主要证明。
通用症状域的`conflicting`在主要presence分析中排除，并分别按非当前阳性和当前阳性
编码进行两项敏感性分析。第二阶段另报告无来源支持证据率及预定错误类型。
"""


def _readme_text() -> str:
    return """# D-P人工效度验证packet 3

本版本替代尚未启动正式评分的packet 2工作流，但不修改或删除packet 2。第一阶段
材料不含任何C5证据或D-P值。给每名评价者发放`blind_cases.csv`、
`blind_case_lines.csv`、`stage1_codebook.md`、`stage1_protocol.md`及其对应的
`reviewer_*_stage1_blank.csv`。先使用`training_*`材料完成培训并冻结码本。

主30人、稀有域富集样本和ID 385的映射只存在于ignored private owner目录。
第一阶段人工共识完成并冻结前，不得生成或发放第二阶段C5证据表。
"""


def _empty_stage2_evidence_schema() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "reviewer_id",
            "blind_id",
            "domain",
            "dp_evidence_id",
            "dp_quote",
            "dp_line_number",
            "dp_polarity",
            "source_supported",
            "speaker_correct",
            "domain_correct",
            "negation_preserved",
            "temporality_correct",
            "subject_correct",
            "context_preserved",
            "duplicate_of_evidence_id",
            "human_evidence_match",
            "error_notes",
        ]
    )


def _empty_stage2_omission_schema() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "reviewer_id",
            "blind_id",
            "domain",
            "human_evidence_id",
            "human_evidence_quote",
            "human_line_number",
            "same_cell_dp_evidence_ids",
            "covered_by_dp",
            "matched_dp_evidence_ids",
            "important_omission",
            "omission_reason",
        ]
    )


def _zip_training_packet(
    *, public_packet: Path, private_packet: Path, reviewer: str
) -> Path:
    output = private_packet / f"reviewer_{reviewer}_training_packet_v3.zip"
    members = [
        "stage1_codebook.md",
        "stage1_protocol.md",
        "training_cases.csv",
        "training_case_lines.csv",
        f"training_reviewer_{reviewer}_stage1_blank.csv",
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(public_packet / name, arcname=name)
    return output


def prepare(
    *,
    public_packet: Path = DEFAULT_PUBLIC_PACKET,
    private_packet: Path = DEFAULT_PRIVATE_PACKET,
    reviewed_spans_path: Path = DEFAULT_REVIEWED_SPANS_PATH,
    force: bool = False,
) -> dict[str, object]:
    if public_packet.exists() and any(public_packet.iterdir()) and not force:
        raise FileExistsError(f"public packet is not empty: {public_packet}")
    if private_packet.exists() and any(private_packet.iterdir()) and not force:
        raise FileExistsError(f"private packet is not empty: {private_packet}")
    public_packet.mkdir(parents=True, exist_ok=True)
    private_packet.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(PARTICIPANTS_PATH)
    domain_counts = pd.read_csv(DOMAIN_COUNT_PATH)
    main_packet2_key_path = PACKET2_PRIVATE / "blind_id_key.csv"
    training_packet2_key_path = PACKET2_PRIVATE / "training_id_key.csv"
    main_packet2 = pd.read_csv(main_packet2_key_path)
    training_packet2 = pd.read_csv(training_packet2_key_path)
    main_ids = set(main_packet2["participant_id"].astype(int))
    training_ids = set(training_packet2["participant_id"].astype(int))

    enrichment = select_enrichment_sample(
        domain_counts,
        excluded_participant_ids={*main_ids, *training_ids, SENTINEL_PARTICIPANT_ID},
        target_quota=ENRICHMENT_TARGET_QUOTA,
        min_n=ENRICHMENT_MIN_N,
        max_n=ENRICHMENT_MAX_N,
    )
    validation_sample = build_validation_sample(
        main_packet2,
        enrichment,
        sentinel_participant_id=SENTINEL_PARTICIPANT_ID,
    )
    blind_key = make_opaque_blind_key(validation_sample, seed=BLIND_ID_SEED)
    blind_cases = build_blind_cases(participants, blind_key)
    blind_lines = build_line_index(blind_cases)
    reviewer_a = build_stage1_review_form(
        blind_cases["blind_id"], reviewer_id="A", order_seed=REVIEWER_A_ORDER_SEED
    )
    reviewer_b = build_stage1_review_form(
        blind_cases["blind_id"], reviewer_id="B", order_seed=REVIEWER_B_ORDER_SEED
    )
    validate_stage1_reviewer_form(
        reviewer_a, expected_blind_ids=blind_cases["blind_id"], require_completed=False
    )
    validate_stage1_reviewer_form(
        reviewer_b, expected_blind_ids=blind_cases["blind_id"], require_completed=False
    )

    # Retain the same five packet-2 training participants but use the revised form.
    training_key = training_packet2[["blind_id", "participant_id"]].copy()
    training_cases = build_blind_cases(participants, training_key)
    training_lines = build_line_index(training_cases)
    training_a = build_stage1_review_form(
        training_cases["blind_id"], reviewer_id="A", order_seed=REVIEWER_A_ORDER_SEED + 1
    )
    training_b = build_stage1_review_form(
        training_cases["blind_id"], reviewer_id="B", order_seed=REVIEWER_B_ORDER_SEED + 1
    )

    # Reviewer-facing stage-1 files. No sample type, IDs, labels, D-P or C5 evidence.
    _write_csv(blind_cases, public_packet / "blind_cases.csv")
    _write_csv(blind_lines, public_packet / "blind_case_lines.csv")
    _write_csv(reviewer_a, public_packet / "reviewer_A_stage1_blank.csv")
    _write_csv(reviewer_b, public_packet / "reviewer_B_stage1_blank.csv")
    _write_csv(training_cases, public_packet / "training_cases.csv")
    _write_csv(training_lines, public_packet / "training_case_lines.csv")
    _write_csv(training_a, public_packet / "training_reviewer_A_stage1_blank.csv")
    _write_csv(training_b, public_packet / "training_reviewer_B_stage1_blank.csv")
    _write_csv(_empty_stage2_evidence_schema(), public_packet / "stage2_evidence_review_schema.csv")
    _write_csv(_empty_stage2_omission_schema(), public_packet / "stage2_omission_review_schema.csv")

    flags = domain_counts[["participant_id"]].copy()
    for target, (domain, state) in RARE_TARGETS.items():
        values = pd.to_numeric(domain_counts[domain], errors="raise")
        flags[target] = values.gt(0) if state == "positive" else values.eq(0)
    main_flags = flags.loc[flags["participant_id"].astype(int).isin(main_ids)]
    enrichment_summary_rows = []
    for target in RARE_TARGETS:
        enrichment_summary_rows.append(
            {
                "rare_target": target,
                "main_sample_target_n": int(main_flags[target].sum()),
                "enriched_sample_target_n": int(enrichment[target].sum()),
                "combined_target_n_descriptive_only": int(
                    main_flags[target].sum() + enrichment[target].sum()
                ),
                "primary_and_enriched_must_be_reported_separately": True,
            }
        )
    enrichment_summary = pd.DataFrame(enrichment_summary_rows)
    _write_csv(enrichment_summary, public_packet / "enrichment_design_summary.csv")

    (public_packet / "stage1_codebook.md").write_text(_codebook_text(), encoding="utf-8")
    (public_packet / "stage1_protocol.md").write_text(
        _protocol_text(len(main_packet2), len(enrichment), len(blind_cases)), encoding="utf-8"
    )
    (public_packet / "README.md").write_text(_readme_text(), encoding="utf-8")

    # Owner-only mapping and D-P reference.
    _write_csv(blind_key, private_packet / "blind_id_key.csv")
    _write_csv(training_key, private_packet / "training_id_key.csv")
    _write_csv(enrichment, private_packet / "enrichment_selection_log.csv")
    scoring_reference = blind_key[["blind_id", "participant_id", "sample_type"]].merge(
        domain_counts[["participant_id", *DOMAIN_FEATURES]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if scoring_reference[list(DOMAIN_FEATURES)].isna().any().any():
        raise ValueError("owner scoring reference is missing a D-P domain count")
    _write_csv(scoring_reference, private_packet / "scoring_reference_owner_only.csv")

    training_zip_a = _zip_training_packet(
        public_packet=public_packet, private_packet=private_packet, reviewer="A"
    )
    training_zip_b = _zip_training_packet(
        public_packet=public_packet, private_packet=private_packet, reviewer="B"
    )

    public_files = [
        path
        for path in sorted(public_packet.iterdir())
        if path.is_file() and path.name not in {"packet_manifest.json", "blind_audit_manifest.json"}
    ]
    manifest: dict[str, object] = {
        "status": "stage1_ready_not_dispatched",
        "analysis": "d_p_blinded_manual_quality_audit_v3",
        "main_sample_n": int((blind_key["sample_type"] == "main").sum()),
        "enriched_sample_n": int((blind_key["sample_type"] == "enriched").sum()),
        "sentinel_sample_n": int((blind_key["sample_type"] == "sentinel").sum()),
        "stage1_validation_n": int(len(blind_cases)),
        "stage1_cells_per_reviewer": int(len(reviewer_a)),
        "training_sample_n": int(len(training_cases)),
        "training_excluded_from_all_validation_statistics": True,
        "main_sample_preserved_from_packet_2": True,
        "enriched_sample_reported_separately": True,
        "sentinel_excluded_from_aggregate_statistics": True,
        "stage2_status": "locked_until_human_consensus_freeze",
        "sampling": {
            "blind_id_seed": BLIND_ID_SEED,
            "enrichment_target_quota": ENRICHMENT_TARGET_QUOTA,
            "enrichment_min_n": ENRICHMENT_MIN_N,
            "enrichment_max_n": ENRICHMENT_MAX_N,
            "rare_targets": list(RARE_TARGETS),
        },
        "inputs": {
            "participant_speech": {
                "path": str(PARTICIPANTS_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256(PARTICIPANTS_PATH),
            },
            "domain_count": {
                "path": str(DOMAIN_COUNT_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256(DOMAIN_COUNT_PATH),
            },
            "packet_2_main_key_owner_only": {"sha256": _sha256(main_packet2_key_path)},
            "packet_2_training_key_owner_only": {"sha256": _sha256(training_packet2_key_path)},
            "reviewed_c5_spans_stage2_owner_input": {
                "restricted": True,
                "filename": reviewed_spans_path.name,
                "sha256": _sha256(reviewed_spans_path),
                "row_n": int(len(pd.read_csv(reviewed_spans_path, usecols=["participant_id"]))),
            },
        },
        "private_owner_file_hashes": {
            path.name: _sha256(path)
            for path in sorted(private_packet.iterdir())
            if path.is_file()
        },
        "training_packet_hashes": {
            training_zip_a.name: _sha256(training_zip_a),
            training_zip_b.name: _sha256(training_zip_b),
        },
        "public_output_hashes": {path.name: _sha256(path) for path in public_files},
        "code_commit": _code_commit(),
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (public_packet / "packet_manifest.json").write_text(manifest_text, encoding="utf-8")
    (public_packet / "blind_audit_manifest.json").write_text(manifest_text, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare packet_3 D-P blinded audit materials")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--public-packet", type=Path, default=DEFAULT_PUBLIC_PACKET)
    parser.add_argument("--private-packet", type=Path, default=DEFAULT_PRIVATE_PACKET)
    parser.add_argument("--reviewed-spans", type=Path, default=DEFAULT_REVIEWED_SPANS_PATH)
    args = parser.parse_args()
    manifest = prepare(
        public_packet=args.public_packet,
        private_packet=args.private_packet,
        reviewed_spans_path=args.reviewed_spans,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "main_sample_n": manifest["main_sample_n"],
                "enriched_sample_n": manifest["enriched_sample_n"],
                "sentinel_sample_n": manifest["sentinel_sample_n"],
                "public_packet": str(args.public_packet),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
