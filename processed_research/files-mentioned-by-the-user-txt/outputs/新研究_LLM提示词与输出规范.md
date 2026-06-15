# 新研究 LLM 提示词与输出规范

## 1. 症状证据抽取 Prompt

用途：从 Participant-only 文本中抽取患者自己的症状证据，用于构造 C5 Symptom evidence-only。

```text
You are a clinical text analysis assistant.

Extract only participant statements that indicate depressive symptoms or functional impairment.

Categories:
1. depressed mood
2. anhedonia
3. sleep problems
4. fatigue or low energy
5. appetite or weight change
6. self-blame or worthlessness
7. concentration problems
8. psychomotor change
9. suicidal ideation or self-harm
10. social or occupational impairment
11. negative evidence, such as explicit denial of symptoms

Rules:
- Use only the participant's own words.
- Do not include the interviewer's questions.
- Keep the original wording as much as possible.
- If no evidence is found for a category, return an empty list.
- Return valid JSON only.

Output format:
{
  "depressed_mood": [],
  "anhedonia": [],
  "sleep": [],
  "fatigue": [],
  "appetite": [],
  "self_worth": [],
  "concentration": [],
  "psychomotor": [],
  "suicidal_ideation": [],
  "impairment": [],
  "negative_evidence": []
}

Participant text:
[TEXT]
```

## 2. LLM 抑郁二分类 Prompt

用途：对五种输入条件统一做零样本分类，避免 prompt 差异造成额外偏差。

```text
You are a clinical text analysis assistant.

Task:
Determine whether the participant is likely to have clinically significant depressive symptoms based on the given interview text.

Criterion:
0 = no clinically significant depression
1 = clinically significant depression

Important rules:
- Base your judgment only on the provided text.
- If the text contains only interviewer questions, do not assume that the participant has the symptom unless the question itself strongly indicates a targeted clinical follow-up.
- Return evidence from the text that supports your decision.
- Return valid JSON only.

Output format:
{
  "label": 0,
  "confidence": 0.00,
  "evidence": [
    {
      "speaker": "participant/interviewer/unknown",
      "quote": "...",
      "evidence_type": "symptom/interviewer_prompt/functional_impairment/negative_evidence/other"
    }
  ],
  "reason": "..."
}

Interview text:
[TEXT]
```

## 3. JSONL 输入格式

每一行一个模型调用任务：

```json
{
  "task_id": "300_full_gpt4omini",
  "participant_id": "300",
  "model": "gpt-4o-mini",
  "input_condition": "full_dialogue",
  "text": "Ellie: ...\nParticipant: ..."
}
```

建议字段：

- `task_id`
- `participant_id`
- `model`
- `input_condition`
- `text`
- `prompt_version`
- `created_at`

## 4. JSONL 输出格式

每一行保存一次模型原始输出和解析结果：

```json
{
  "task_id": "300_full_gpt4omini",
  "participant_id": "300",
  "model": "gpt-4o-mini",
  "input_condition": "full_dialogue",
  "true_label": 1,
  "raw_response": "{...}",
  "parsed": {
    "label": 1,
    "confidence": 0.83,
    "evidence": [
      {
        "speaker": "participant",
        "quote": "I have been feeling down lately.",
        "evidence_type": "symptom"
      }
    ],
    "reason": "..."
  },
  "parse_ok": true,
  "error": null
}
```

## 5. Evidence 长表格式

把 LLM 输出展开成一条 evidence 一行：

| 字段 | 含义 |
|---|---|
| participant_id | 被试 ID |
| model | 模型名称 |
| input_condition | 输入条件 |
| true_label | 真实标签 |
| pred_label | 模型预测 |
| evidence_index | 第几条 evidence |
| speaker | participant / interviewer / unknown |
| quote | 模型引用的证据原文 |
| evidence_type | symptom / interviewer_prompt / functional_impairment / negative_evidence / other |
| valid_by_rule | 规则初筛是否有效，可后续人工覆盖 |

## 6. Evidence 指标计算规则

EVR：

```text
EVR = 有效患者症状证据数 / 全部证据数
```

IER：

```text
IER = 访谈者问题证据数 / 全部证据数
```

建议 operational definition：

- 有效患者症状证据：`speaker == participant` 且 `evidence_type` 属于 `symptom`, `functional_impairment`, `negative_evidence`。
- 访谈者问题证据：`speaker == interviewer` 或 `evidence_type == interviewer_prompt`。
- 无效证据：无法在输入文本中定位、与判断无关、或只是泛泛解释。

## 7. 失败处理

如果模型没有输出合法 JSON：

1. 保留 `raw_response`。
2. `parse_ok = false`。
3. 不要手动补标签。
4. 用同一 prompt 重试一次，并记录 `retry_index = 1`。
5. 第二次仍失败，纳入 failure rate 报告。
