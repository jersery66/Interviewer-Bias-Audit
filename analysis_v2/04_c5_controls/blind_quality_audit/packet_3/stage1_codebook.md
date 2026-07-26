# D-P人工效度验证第一阶段码本（packet 3）

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
