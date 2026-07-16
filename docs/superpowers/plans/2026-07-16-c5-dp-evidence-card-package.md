# C5/D-P 简化证据卡片评分包实施计划

> **供执行代理使用：** 必须逐项执行本计划，并使用 `subagent-driven-development`（推荐）或 `executing-plans`。所有步骤使用复选框追踪。

**目标：** 从锁定的 1,137 条 C5/D-P 证据中，生成两名评分者可在 45 至 60 分钟内完成的中文盲法证据卡片培训包和正式评分包。

**架构：** Python 模块负责冻结输入核验、去重、均衡抽样、上下文构建、盲法表和负责人对照表；一个独立 JavaScript 构建器使用 `@oai/artifact-tool` 将经过验证的 JSON 表转换为中文 Excel。评分者材料与负责人材料物理隔离，正式工作簿先保存在负责人目录，待培训和编码手册冻结后再发放。

**技术栈：** Python 3、pandas、pytest、JavaScript、`@oai/artifact-tool`、JSON/CSV、SHA-256。

---

## 文件结构

- 新建 `reanalysis_v2/c5_evidence_card_validation.py`：冻结输入、上下文、抽样、盲法表、负责人表和清单逻辑。
- 新建 `scripts/build_c5_evidence_card_validation.py`：命令行入口，只协调输入、输出和失败清理。
- 新建 `scripts/build_c5_evidence_card_workbooks.mjs`：唯一 Excel 构建器，读取已验证 JSON 并输出四个工作簿。
- 新建 `tests/test_c5_evidence_card_validation.py`：纯 Python 单元测试和端到端合成数据测试。
- 新建 `analysis_v2/04_c5_controls/blind_quality_audit/evidence_card_v1/README.md`：中文公开说明，不含引文、参与者映射或评分材料。
- 生成但不纳入 Git：评分者工作簿、负责人对照表、抽样审计、清单和哈希文件。

本计划只生成评分材料，不计算尚未产生的人工一致性结果。人工结果回收后的统计计算另设实施计划。

### 任务一：冻结输入和上下文构建

**文件：**

- 新建：`reanalysis_v2/c5_evidence_card_validation.py`
- 新建：`tests/test_c5_evidence_card_validation.py`

- [ ] **步骤1：编写冻结输入失败测试**

在测试文件中创建最小证据表和文本映射，并验证错误哈希、缺列、未知域和无法定位引文时立即失败：

```python
from pathlib import Path

import pandas as pd
import pytest

from reanalysis_v2.c5_evidence_card_validation import (
    C5_CANONICAL_SHA256,
    DOMAIN_NAMES,
    load_and_validate_spans,
    locate_quote_context,
)


def test_frozen_input_hash_must_match(tmp_path: Path) -> None:
    path = tmp_path / "spans.csv"
    path.write_text("participant_id,domain,polarity,exact_quote,source_order\n1,depressed_mood,present,sad,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_validate_spans(path, expected_sha256=C5_CANONICAL_SHA256)


def test_unknown_domain_is_rejected() -> None:
    frame = pd.DataFrame({
        "participant_id": [1],
        "domain": ["unknown_domain"],
        "polarity": ["present"],
        "exact_quote": ["sad"],
        "source_order": [1],
    })
    assert "unknown_domain" not in DOMAIN_NAMES
    with pytest.raises(ValueError, match="未知症状域"):
        load_and_validate_spans(frame, expected_sha256=None)


def test_quote_must_be_locatable_in_participant_text() -> None:
    with pytest.raises(ValueError, match="无法定位证据"):
        locate_quote_context("this text has no match", "different quote")
```

- [ ] **步骤2：运行测试并确认失败**

运行：

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k "frozen_input or unknown_domain or quote_must" -q
```

预期：因新模块尚不存在而失败。

- [ ] **步骤3：实现常量、哈希核验和输入验证**

在新模块中定义锁定路径、哈希和十个域，并实现以下接口：

```python
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

C5_CANONICAL_RELATIVE = Path(
    "processed_research/latest_usable_data_official142_20260702/"
    "04_c5_evidence_sources/c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv"
)
C5_CANONICAL_SHA256 = "794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e"
PARTICIPANT_TEXT_RELATIVE = Path("processed_research/participant_for_evidence_extraction.jsonl")
PARTICIPANT_TEXT_SHA256 = "f64eefd1b9560a845a4e838b5260016fbad840a2c957ef048c5feea965e4b549"
DOMAIN_NAMES = (
    "anhedonia_interest",
    "appetite_weight",
    "concentration_psychomotor",
    "depressed_mood",
    "functioning_impairment",
    "mental_health_history",
    "protective_or_absent_symptom",
    "self_worth_guilt",
    "sleep_fatigue_energy",
    "suicide_self_harm",
)
REQUIRED_SPAN_COLUMNS = {
    "participant_id", "domain", "polarity", "exact_quote", "source_order"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_spans(
    source: Path | pd.DataFrame,
    *,
    expected_sha256: str | None,
) -> pd.DataFrame:
    if isinstance(source, Path):
        if expected_sha256 is not None and sha256_file(source) != expected_sha256:
            raise ValueError("C5冻结输入SHA-256不匹配")
        frame = pd.read_csv(source)
    else:
        frame = source.copy()
    missing = sorted(REQUIRED_SPAN_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"C5证据表缺少列：{missing}")
    unknown = sorted(set(frame["domain"].astype(str)).difference(DOMAIN_NAMES))
    if unknown:
        raise ValueError(f"发现未知症状域：{unknown}")
    frame["participant_id"] = pd.to_numeric(frame["participant_id"], errors="raise").astype(int)
    frame["source_order"] = pd.to_numeric(frame["source_order"], errors="raise").astype(int)
    if frame["exact_quote"].isna().any() or frame["exact_quote"].astype(str).str.strip().eq("").any():
        raise ValueError("C5证据表存在空引文")
    return frame.reset_index(drop=True)
```

- [ ] **步骤4：实现参与者文本加载、去标识和最小上下文**

实现逐行读取 `processed_research/participant_for_evidence_extraction.jsonl`，并用规范化空白后的字符串定位证据。上下文仅取证据前后各最多 180 个字符：

```python
def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def load_participant_texts(
    path: Path,
    *,
    expected_sha256: str = PARTICIPANT_TEXT_SHA256,
) -> dict[int, str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("参与者上下文输入SHA-256不匹配")
    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            participant_id = int(row["participant_id"])
            if participant_id in result:
                raise ValueError(f"参与者文本重复：{participant_id}")
            result[participant_id] = str(row["text"])
    return result


def deidentify_text(value: object) -> str:
    text = normalize_space(value)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,2}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"(?i)\b(my name is|i am|i'm)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", r"\1 [NAME]", text)
    return re.sub(r"\b\d{3,}\b", "[NUMBER]", text)


def locate_quote_context(text: str, quote: str, *, flank_chars: int = 180) -> tuple[str, str, str]:
    normalized_text = normalize_space(text)
    normalized_quote = normalize_space(quote)
    start = normalized_text.casefold().find(normalized_quote.casefold())
    if start < 0:
        raise ValueError(f"无法定位证据：{normalized_quote[:80]}")
    end = start + len(normalized_quote)
    before = normalized_text[max(0, start - flank_chars):start].strip()
    after = normalized_text[end:min(len(normalized_text), end + flank_chars)].strip()
    return deidentify_text(before), deidentify_text(normalized_quote), deidentify_text(after)
```

- [ ] **步骤5：运行定向测试**

运行：

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k "frozen_input or unknown_domain or quote_must" -q
```

预期：全部通过。

- [ ] **步骤6：提交任务一**

```powershell
git add reanalysis_v2/c5_evidence_card_validation.py tests/test_c5_evidence_card_validation.py
git commit -m "feat: validate frozen C5 evidence-card inputs"
```

### 任务二：均衡抽样和盲法表

**文件：**

- 修改：`reanalysis_v2/c5_evidence_card_validation.py`
- 修改：`tests/test_c5_evidence_card_validation.py`

- [ ] **步骤1：编写抽样约束测试**

使用每域 16 条、不同参与者的合成数据，验证正式样本每域 12 条、培训样本每域 1 条、无记录重叠、每名参与者正式样本最多 2 条、固定种子可复现：

```python
def synthetic_balanced_spans() -> pd.DataFrame:
    rows = []
    for domain_index, domain in enumerate(DOMAIN_NAMES):
        for item_index in range(16):
            rows.append({
                "participant_id": 1000 + domain_index * 16 + item_index,
                "domain": domain,
                "polarity": "present",
                "exact_quote": f"quote {domain_index} {item_index}",
                "source_order": item_index + 1,
            })
    return pd.DataFrame(rows)


def test_balanced_sampling_contract() -> None:
    first = select_evidence_cards(synthetic_balanced_spans(), seed=20260716)
    second = select_evidence_cards(synthetic_balanced_spans(), seed=20260716)
    pd.testing.assert_frame_equal(first.training, second.training)
    pd.testing.assert_frame_equal(first.formal, second.formal)
    assert first.training.groupby("domain").size().eq(1).all()
    assert first.formal.groupby("domain").size().eq(12).all()
    assert len(first.training) == 10
    assert len(first.formal) == 120
    assert set(first.training["candidate_key"]).isdisjoint(first.formal["candidate_key"])
    assert first.formal.groupby("participant_id").size().max() <= 2
```

再加入不足 13 条的域并确认硬失败：

```python
def test_domain_with_fewer_than_thirteen_candidates_stops() -> None:
    frame = synthetic_balanced_spans()
    domain = DOMAIN_NAMES[-1]
    frame = frame.loc[~((frame["domain"] == domain) & (frame["source_order"] > 12))]
    with pytest.raises(ValueError, match="至少需要13条"):
        select_evidence_cards(frame, seed=20260716)
```

- [ ] **步骤2：运行测试并确认失败**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k "balanced_sampling or fewer_than_thirteen" -q
```

预期：因 `select_evidence_cards` 尚不存在而失败。

- [ ] **步骤3：实现候选去重和确定性约束抽样**

实现 `EvidenceCardSample` 数据类。先按参与者、域和规范化引文去重；培训样本每域取 1 条；正式样本按候选数从少到多处理各域，每域取 12 条，并限制每名参与者最多 2 条。为了避免单次贪心排序误判不可行，使用固定种子进行最多 2,000 次确定性重排；第一个满足约束的解即为锁定解：

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvidenceCardSample:
    training: pd.DataFrame
    formal: pd.DataFrame
    audit: pd.DataFrame


def _candidate_key(row: pd.Series) -> str:
    payload = f"{int(row.participant_id)}|{row.domain}|{normalize_space(row.exact_quote).casefold()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_evidence_cards(spans: pd.DataFrame, *, seed: int) -> EvidenceCardSample:
    frame = spans.copy()
    frame["normalized_quote"] = frame["exact_quote"].map(normalize_space).str.casefold()
    frame = frame.sort_values(["domain", "participant_id", "source_order"], kind="stable")
    frame = frame.drop_duplicates(["participant_id", "domain", "normalized_quote"], keep="first")
    frame["candidate_key"] = frame.apply(_candidate_key, axis=1)
    counts = frame.groupby("domain").size().reindex(DOMAIN_NAMES, fill_value=0)
    too_small = counts[counts < 13]
    if not too_small.empty:
        raise ValueError(f"每个域至少需要13条合格证据：{too_small.to_dict()}")

    domain_order = sorted(DOMAIN_NAMES, key=lambda name: (int(counts[name]), name))
    for attempt in range(2000):
        rng = np.random.default_rng(seed + attempt)
        training_parts = []
        formal_parts = []
        participant_uses: dict[int, int] = {}
        feasible = True
        for domain in domain_order:
            candidates = frame.loc[frame["domain"].eq(domain)].copy()
            candidates["_random"] = rng.random(len(candidates))
            candidates = candidates.sort_values(["_random", "participant_id", "source_order"], kind="stable")
            training = candidates.iloc[[0]].copy()
            remaining = candidates.iloc[1:].copy()
            selected = []
            for index, row in remaining.iterrows():
                participant_id = int(row["participant_id"])
                if participant_uses.get(participant_id, 0) >= 2:
                    continue
                selected.append(index)
                participant_uses[participant_id] = participant_uses.get(participant_id, 0) + 1
                if len(selected) == 12:
                    break
            if len(selected) != 12:
                feasible = False
                break
            training_parts.append(training)
            formal_parts.append(remaining.loc[selected].copy())
        if feasible:
            training = pd.concat(training_parts, ignore_index=True).drop(columns="_random")
            formal = pd.concat(formal_parts, ignore_index=True).drop(columns="_random")
            audit = counts.rename("eligible_n").reset_index(names="domain")
            audit["training_n"] = 1
            audit["formal_n"] = 12
            audit["seed"] = seed
            audit["attempt"] = attempt
            return EvidenceCardSample(training=training, formal=formal, audit=audit)
    raise RuntimeError("在固定约束下未找到120条正式证据的可行抽样")
```

- [ ] **步骤4：编写盲法字段和随机行序测试**

```python
def test_reviewer_tables_are_blinded_and_have_different_orders() -> None:
    sample = select_evidence_cards(synthetic_balanced_spans(), seed=20260716)
    texts = {int(pid): f"before {quote} after" for pid, quote in zip(sample.formal["participant_id"], sample.formal["exact_quote"])}
    owner, reviewer_a, reviewer_b = build_review_tables(sample.formal, texts, seed=20260716)
    prohibited = {"participant_id", "model_domain", "model_polarity", "label", "source_order"}
    assert prohibited.isdisjoint(reviewer_a.columns)
    assert set(reviewer_a["review_case_id"]) == set(reviewer_b["review_case_id"])
    assert reviewer_a["review_case_id"].tolist() != reviewer_b["review_case_id"].tolist()
    assert owner["review_case_id"].is_unique
    answer_fields = ["human_span_valid", "human_domain", "human_polarity", "notes"]
    assert reviewer_a[answer_fields].fillna("").eq("").all().all()
    assert reviewer_b[answer_fields].fillna("").eq("").all().all()
```

- [ ] **步骤5：实现负责人表和评分者表**

评分者表固定为八列，负责人表保留模型字段和抽样来源。`review_case_id` 由冻结哈希、候选键和样本类型生成，不含可逆参与者编号：

```python
REVIEWER_COLUMNS = (
    "review_case_id", "evidence_quote", "context_before", "context_after",
    "human_span_valid", "human_domain", "human_polarity", "notes",
)
PROHIBITED_REVIEWER_COLUMNS = {
    "participant_id", "label", "paper_label_phq8_ge10", "paper_phq8_score",
    "model_domain", "model_polarity", "domain", "polarity", "source_order",
}


def _review_case_id(candidate_key: str, scope: str) -> str:
    digest = hashlib.sha256(f"{C5_CANONICAL_SHA256}|{scope}|{candidate_key}".encode("utf-8")).hexdigest()
    return f"C5-{scope.upper()}-{digest[:14]}"


def build_review_tables(
    selected: pd.DataFrame,
    participant_texts: dict[int, str],
    *,
    seed: int,
    scope: str = "formal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected.itertuples(index=False):
        before, quote, after = locate_quote_context(participant_texts[int(row.participant_id)], row.exact_quote)
        rows.append({
            "review_case_id": _review_case_id(row.candidate_key, scope),
            "participant_id": int(row.participant_id),
            "source_order": int(row.source_order),
            "candidate_key": row.candidate_key,
            "model_domain": row.domain,
            "model_polarity": row.polarity,
            "model_exact_quote": row.exact_quote,
            "evidence_quote": quote,
            "context_before": before,
            "context_after": after,
            "sample_scope": scope,
        })
    owner = pd.DataFrame(rows)
    reviewer = owner[["review_case_id", "evidence_quote", "context_before", "context_after"]].copy()
    for field in ("human_span_valid", "human_domain", "human_polarity", "notes"):
        reviewer[field] = ""
    reviewer = reviewer.loc[:, REVIEWER_COLUMNS]
    if PROHIBITED_REVIEWER_COLUMNS.intersection(reviewer.columns):
        raise AssertionError("评分者表包含禁止字段")
    reviewer_a = reviewer.sample(frac=1, random_state=seed).reset_index(drop=True)
    reviewer_b = reviewer.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    if reviewer_a["review_case_id"].tolist() == reviewer_b["review_case_id"].tolist():
        raise AssertionError("A、B行序不得相同")
    return owner, reviewer_a, reviewer_b
```

- [ ] **步骤6：运行任务二测试并提交**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k "sampling or reviewer_tables" -q
git add reanalysis_v2/c5_evidence_card_validation.py tests/test_c5_evidence_card_validation.py
git commit -m "feat: sample balanced blinded C5 evidence cards"
```

### 任务三：原子化生成中间表、清单和中文说明

**文件：**

- 修改：`reanalysis_v2/c5_evidence_card_validation.py`
- 新建：`scripts/build_c5_evidence_card_validation.py`
- 修改：`tests/test_c5_evidence_card_validation.py`
- 新建：`analysis_v2/04_c5_controls/blind_quality_audit/evidence_card_v1/README.md`

- [ ] **步骤1：编写输出隔离和失败清理测试**

测试输出根目录只包含三个隔离区：`评分者A_培训`、`评分者B_培训`、`OWNER_ONLY`。正式评分输入 JSON 只能位于 `OWNER_ONLY/待编码手册冻结后发放`。任何预检失败时输出根目录不得存在：

```python
def test_package_layout_separates_reviewer_and_owner_files(tmp_path: Path) -> None:
    result = write_evidence_card_intermediates(
        synthetic_package_input(),
        output_root=tmp_path / "package",
        source_hash="source-hash",
        seed=20260716,
    )
    assert result.reviewer_a_training.parent.name == "评分者A_培训"
    assert result.reviewer_b_training.parent.name == "评分者B_培训"
    assert "OWNER_ONLY" in result.owner_crosswalk.parts
    assert "待编码手册冻结后发放" in result.reviewer_a_formal.parts
    assert not list((tmp_path / "package" / "评分者A_培训").glob("*formal*"))


def test_preflight_failure_leaves_no_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "package"
    with pytest.raises(ValueError):
        build_evidence_card_package(
            canonical_path=tmp_path / "missing.csv",
            participant_text_path=tmp_path / "missing.jsonl",
            output_root=output,
            seed=20260716,
        )
    assert not output.exists()
```

- [ ] **步骤2：运行测试并确认失败**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k "package_layout or preflight_failure" -q
```

- [ ] **步骤3：实现暂存目录、JSON中间表和清单**

所有数据先写到同级 `.<目录名>.staging`，完成哈希和内容检查后再原子重命名。中间表使用 UTF-8 JSON，避免 CSV 引文换行产生歧义。清单必须记录：输入哈希、种子、每域数量、参与者最大复用数、培训／正式交集、A/B集合及行序检查、每个输出文件 SHA-256。

公开评分者 README 只解释三个判断字段；负责人 README 明确正式工作簿不得在编码手册冻结前发放。所有 Markdown 使用中文。

- [ ] **步骤4：实现CLI**

命令行参数固定如下：

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成C5/D-P简化证据卡片评分包")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args(argv)
```

CLI 在任何目录创建前依次完成：冻结证据哈希核验、参与者上下文哈希核验、1137 条证据计数核验、十域候选数量核验和全部选中引文定位核验。

- [ ] **步骤5：新增中文公开说明**

`analysis_v2/04_c5_controls/blind_quality_audit/evidence_card_v1/README.md` 必须说明：

- 本版替代旧的全文十域评分工作流；
- 每位评分者正式审核 120 张证据卡；
- 只审核有效性、域和极性；
- 不检查遗漏证据，不估计召回率；
- 评分者材料受限，不纳入 Git；
- 当前目录只记录设计和生成方式，不代表人工审核已经完成。

- [ ] **步骤6：运行测试、编译并提交**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -q
py -3 -m py_compile reanalysis_v2/c5_evidence_card_validation.py scripts/build_c5_evidence_card_validation.py
git diff --check
git add reanalysis_v2/c5_evidence_card_validation.py scripts/build_c5_evidence_card_validation.py tests/test_c5_evidence_card_validation.py analysis_v2/04_c5_controls/blind_quality_audit/evidence_card_v1/README.md
git commit -m "feat: prepare restricted C5 evidence-card package inputs"
```

### 任务四：使用 artifact-tool 生成四个中文Excel工作簿

**文件：**

- 新建：`scripts/build_c5_evidence_card_workbooks.mjs`
- 修改：`tests/test_c5_evidence_card_validation.py`

- [ ] **步骤1：核验Excel运行时**

调用工作区依赖加载器，确认返回的 `node_modules` 中可解析 `@oai/artifact-tool`。在会话临时目录创建到该 `node_modules` 的目录联接。若加载器或依赖不可用，立即停止并报告；不得改用 `openpyxl`、`xlsxwriter`、`pandas.ExcelWriter` 或自行寻找其他包。

- [ ] **步骤2：编写工作簿合同测试**

端到端测试使用合成 JSON 调用构建器，并检查四个 XLSX 均存在、工作表名称正确、120 行正式评分和 10 行培训评分、答案单元格为空、工作簿压缩包中不出现 `participant_id`、`model_domain`、`model_polarity` 或 PHQ 字段。

```python
def test_generated_workbooks_follow_blinding_contract(tmp_path: Path) -> None:
    inputs = write_synthetic_workbook_inputs(tmp_path / "inputs")
    outputs = run_workbook_builder(inputs, tmp_path / "outputs")
    assert {path.name for path in outputs} == {
        "C5_DP_证据卡片培训_评分者A.xlsx",
        "C5_DP_证据卡片培训_评分者B.xlsx",
        "C5_DP_证据卡片正式评分_评分者A.xlsx",
        "C5_DP_证据卡片正式评分_评分者B.xlsx",
    }
    for path in outputs:
        assert path.stat().st_size > 0
        assert_xlsx_has_no_prohibited_strings(path)
        assert_xlsx_answer_cells_blank(path)
```

- [ ] **步骤3：运行测试并确认失败**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k generated_workbooks -q
```

- [ ] **步骤4：实现唯一JavaScript工作簿构建器**

构建器从命令行读取 `--input-dir` 和 `--output-dir`，使用 `Workbook.create()` 创建工作簿，并用 `SpreadsheetFile.exportXlsx()` 导出。每个工作簿使用统一样式：深蓝标题、浅蓝说明区、白底评分表、冻结表头、隐藏网格线、证据和上下文自动换行。

评分表数据列固定为：审核编号、证据原文、前文、后文、证据是否有效、人工症状域、人工极性、必要备注。后三个判断字段使用下拉框：

```javascript
const validityOptions = ["有效", "无效", "不确定"];
const polarityOptions = ["存在", "否认或保护", "含糊"];
const domainOptions = [
  "anhedonia_interest", "appetite_weight", "concentration_psychomotor",
  "depressed_mood", "functioning_impairment", "mental_health_history",
  "protective_or_absent_symptom", "self_worth_guilt",
  "sleep_fatigue_energy", "suicide_self_harm",
];

scoreSheet.getRange(`E2:E${lastRow}`).dataValidation = {
  rule: { type: "list", values: validityOptions },
};
scoreSheet.getRange(`F2:F${lastRow}`).dataValidation = {
  rule: { type: "list", values: domainOptions },
};
scoreSheet.getRange(`G2:G${lastRow}`).dataValidation = {
  rule: { type: "list", values: polarityOptions },
};
scoreSheet.freezePanes.freezeRows(1);
scoreSheet.freezePanes.freezeColumns(1);
```

为避免评分者误填，证据和上下文列使用浅灰锁定样式，答案列使用浅黄色填充；不依赖颜色编码分析数据。条件格式仅用于提示：选择“有效”但域或极性为空时标红，选择“无效”时域和极性非空也标红。

- [ ] **步骤5：在构建器中加入inspect、错误扫描和渲染验证**

每个工作簿导出前执行：

```javascript
await workbook.inspect({
  kind: "table",
  sheetId: scoreSheet.name,
  range: `A1:H${lastRow}`,
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 8,
});
await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "公式错误扫描",
});
```

分别渲染每个工作表的已用区域到临时 PNG，确认标题、说明、表头和至少两行证据可见。预览图只用于质量检查，不进入交付包。

- [ ] **步骤6：运行工作簿定向测试并提交**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -k generated_workbooks -q
git add scripts/build_c5_evidence_card_workbooks.mjs tests/test_c5_evidence_card_validation.py
git commit -m "feat: build blinded C5 evidence-card workbooks"
```

### 任务五：生成受限评分包并完成全面核验

**文件：**

- 不修改 Git 跟踪的分析结果。
- 在桌面受限目录生成评分包和负责人材料。

- [ ] **步骤1：确认代码工作树和冻结输入**

```powershell
git status --porcelain
git rev-parse HEAD
Get-FileHash -Algorithm SHA256 "F:\数据库\DAIC-WOZ\processed_research\latest_usable_data_official142_20260702\04_c5_evidence_sources\c5_quotes_only_v3_gpt55_reviewed_spans_official142.csv"
Get-FileHash -Algorithm SHA256 "F:\数据库\DAIC-WOZ\processed_research\participant_for_evidence_extraction.jsonl"
```

要求工作树干净；证据输入哈希为 `794d7b43ebd95d57ac4a90a4217953a9bea27e98ff91b13c31b0a06f9aa48c7e`，上下文输入哈希为 `f64eefd1b9560a845a4e838b5260016fbad840a2c957ef048c5feea965e4b549`。

- [ ] **步骤2：生成受限中间表和工作簿**

目标根目录：

```text
C:\Users\Jersery\Desktop\C5-DP人机一致性_简化证据卡片_20260716
```

运行 Python CLI 生成中间表，再使用已核验的 artifact-tool 运行时执行 JavaScript 构建器。评分者A、B外发目录只放培训工作簿和中文README；两份正式工作簿放在：

```text
OWNER_ONLY\待编码手册冻结后发放\
```

- [ ] **步骤3：执行内容合同核验**

逐项核验：

- 冻结输入 1,137 行且哈希匹配；
- 培训样本 10 条、每域 1 条；
- 正式样本 120 条、每域 12 条；
- 培训与正式候选键无交集；
- 正式样本中每名参与者最多 2 条；
- A、B正式审核编号集合相同且行序不同；
- 四个工作簿答案字段全空；
- 下拉选项与中文编码手册一致；
- 评分者工作簿无参与者编号、PHQ、模型域、模型极性或模型状态；
- 负责人表不在评分者目录；
- 所有 Markdown 正文为中文；
- 所有交付文件哈希进入清单。

- [ ] **步骤4：视觉检查四个工作簿的全部工作表**

使用 artifact-tool 的 `render` 输出临时预览，逐张检查：文本无裁切、证据可阅读、输入列清晰、冻结窗格合理、无空白默认工作表、无公式错误。修复只允许调整格式，不得改变抽样或审核编号。

- [ ] **步骤5：运行全部代码验证**

```powershell
py -3 -m pytest tests/test_c5_evidence_card_validation.py -q
py -3 -m pytest -q
py -3 -m py_compile reanalysis_v2/c5_evidence_card_validation.py scripts/build_c5_evidence_card_validation.py
git diff --check
git status --porcelain
```

预期：全部测试通过，工作树干净。

- [ ] **步骤6：提交代码但不提交受限材料**

如果任务四之后仍有代码或中文公开说明修正，只提交代码、测试和公开说明。不得提交任何引文、工作簿、负责人对照表或桌面受限材料：

```powershell
git add reanalysis_v2/c5_evidence_card_validation.py scripts/build_c5_evidence_card_validation.py scripts/build_c5_evidence_card_workbooks.mjs tests/test_c5_evidence_card_validation.py analysis_v2/04_c5_controls/blind_quality_audit/evidence_card_v1/README.md
git commit -m "feat: add simplified C5 evidence-card validation package"
```

最终返回四个工作簿路径、抽样审计摘要、清单哈希、测试结果和明确的发放顺序。不得声称人工审核或人机一致性统计已经完成。
