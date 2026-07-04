# 附录 A：C4 interviewer-cleaned 清洗规则与审计说明

生成日期：2026-06-15

规则版本：`diagnostic_prompt_cleaning_article_aligned_v1_2026-06-12`

源规则文件：`F:\数据库\E-daic\processed_research\interviewer_cleaning_rules_article_aligned.json`

## A1. C4 的研究目的

C4 `interviewer_cleaned` 的目的不是把访谈者影响完全清除，而是在 C3 `interviewer_only` 的基础上删除显性诊断提示问题，检验访谈者话语的预测能力是否主要来自直接症状询问。

因此，C4 应被解释为“诊断性提示降低条件”（diagnostic-prompt reduction condition），而不是“无访谈者偏差条件”。如果 C4 仍然有较高预测力，说明访谈者话语中仍可能存在访谈流程、动态追问、对话结构或互动节奏等间接信号。

## A2. 操作性定义

C4 的构建单位是“访谈者整轮话语”（interviewer turn），不是单词或短语。

具体流程如下：

1. 仅处理 `speaker_norm == interviewer` 的轮次。
2. 对每个访谈者轮次的 `text` 字段运行 8 组正则规则，匹配时使用 `re.IGNORECASE`。
3. 只要某个访谈者轮次命中任意规则，就将该整轮标记为 `is_removed_diagnostic_prompt = True`。
4. C4 文本由未命中规则的访谈者轮次按原始顺序拼接得到。
5. 该步骤不读取 PHQ-8 标签，不读取模型预测结果，也不修改受访者话语。
6. 对于没有访谈者话语的样本，C3/C4 文本为空，并在建模时由基线脚本排除。

## A3. 删除原则

删除的核心标准是：访谈者问题本身是否直接暴露了抑郁筛查、PHQ 风格症状、心理健康诊断史、治疗史或药物史。

删除包括三类情况：

1. 直接询问抑郁核心症状，例如情绪低落、兴趣下降、睡眠、食欲、自杀意念。
2. 直接询问心理健康诊断或治疗史，例如 depression diagnosis、PTSD、anxiety、therapy、medication。
3. 虽然不是 PHQ-8 原句，但具有明显诊断提示作用的心理健康问题，例如 self-harm、psychologist、psychiatrist。

## A4. 保留原则

保留的核心标准是：访谈者话语虽然可能参与访谈流程，但没有直接提供心理健康诊断或症状标签。

保留包括：

1. 寒暄、确认、转场和结束语。
2. 出生地、居住地、教育、工作、家庭、兴趣、日常生活等背景问题。
3. 一般性情绪或生活事件开放问题，但未直接出现症状词或治疗诊断词时保留。
4. 军旅经历等背景问题本身保留；只有 PTSD、诊断、治疗等心理健康词命中时才删除。
5. 不含规则关键词的普通追问保留，即使它可能间接反映访谈流程。

## A5. 八类规则

| rule_id | 删除对象 | 典型触发词或表达 | 判定说明 |
|---|---|---|---|
| `mood_depression` | 抑郁心境、悲伤、绝望、情绪低落 | depressed, depression, sad, down, hopeless, unhappy, feeling down | 删除直接询问低落或抑郁情绪的访谈者轮次。 |
| `anhedonia_interest` | 兴趣或愉悦感下降 | interest, pleasure, enjoy | 删除可能对应 anhedonia 的兴趣/愉悦问题；这类规则较敏感，可能删除部分较宽泛的兴趣问题。 |
| `sleep_fatigue_energy` | 睡眠、疲劳、精力 | sleep, sleep well, good night's sleep, insomnia, tired, fatigue, energy | 删除睡眠质量、疲劳和精力相关提问。 |
| `appetite_weight` | 食欲、进食、体重变化 | appetite, eating, weight | 删除食欲和体重相关提问。 |
| `self_worth_guilt` | 自我价值、内疚、自责、失败感 | worthless, self-worth, blame yourself, failure, feel bad about yourself | 删除与无价值感、内疚和失败感直接相关的提问。 |
| `concentration_psychomotor` | 注意力和精神运动变化 | concentrate, concentration, restless, slowed down, moving or speaking so slowly | 删除注意力、坐立不安或精神运动迟滞相关提问。 |
| `suicide_self_harm` | 自杀意念、自伤、死亡相关想法 | suicide, suicidal, kill yourself, hurt yourself, self-harm, better off dead, disturbing thoughts | 删除自杀、自伤和相关侵入性想法提问。 |
| `mental_health_history` | 心理健康诊断、治疗、咨询、药物史 | mental health, therapy, therapist, counseling, treatment, medication, antidepressant, diagnosis, anxiety, PTSD, psychologist, psychiatrist | 删除诊断、治疗、药物、求助史和心理健康服务相关提问。 |

## A6. 边界案例

| 访谈者问题类型 | C4 处理 | 理由 |
|---|---|---|
| `have you been diagnosed with depression` | 删除 | 同时命中 `mood_depression` 和 `mental_health_history`。 |
| `do you feel down` | 删除 | 直接询问低落情绪，命中 `mood_depression`。 |
| `how easy is it for you to get a good night's sleep` | 删除 | 直接询问睡眠，命中 `sleep_fatigue_energy`。 |
| `have you ever been diagnosed with p_t_s_d` | 删除 | 直接询问心理健康诊断史，命中 `mental_health_history`。 |
| `have you ever served in the military` | 保留 | 军旅经历本身是背景问题，不等同于 PTSD 或心理健康诊断。 |
| `where are you from originally` | 保留 | 背景信息，不含症状或心理健康诊断提示。 |
| `tell me about your job` | 保留 | 工作背景问题，不含诊断性提示。 |
| `what do you like to do for fun` | 通常保留 | 若没有命中 `interest/pleasure/enjoy` 等规则词，则保留；若使用 `enjoy` 等词，则可能被删除。 |

## A7. 审计结果

| 指标 | 数值 |
|---|---:|
| participants | 189 |
| participants_with_removed | 186 |
| removed_turns | 1,384 |
| retained_turns | 13,615 |
| removed_words | 20,107 |
| retained_words | 82,875 |
| mean_removed_turns | 7.32 |
| median_removed_turns | 7.00 |
| mean_turn_removal_ratio | 0.094 |
| mean_word_removal_ratio | 0.194 |

## A8. 人工抽查设计

人工抽查模板为：`processed_research\robustness_outputs\appendix_A_C4_manual_review_template.csv`

模板包含被删除与被保留的访谈者轮次，并保留两个空白字段：

| 字段 | 用途 |
|---|---|
| `human_review_decision` | 人工判定：`correct_remove`、`false_positive_remove`、`correct_retain`、`false_negative_retain`、`unclear`。 |
| `reviewer_notes` | 记录误删、漏删或边界不确定的理由。 |

建议人工抽查时至少覆盖三类样本：

1. `focus_pattern`：规则关注的典型命中和边界案例。
2. `random_removed_nonfocus`：随机抽取的非重点删除轮次，用于估计误删。
3. `random_retained_nonfocus`：随机抽取的非重点保留轮次，用于估计漏删。

## A9. 已知限制

1. 规则是关键词和正则规则，不是人工语义标注，可能存在误删和漏删。
2. C4 不删除所有访谈者互动信号；访谈者的动态追问、问题顺序和访谈流程仍可能携带预测信息。
3. 当前 article-aligned 规则没有单独删除所有 trauma 相关表达；只有 PTSD、anxiety、therapy、diagnosis 等心理健康相关表达会触发删除。
4. `anhedonia_interest` 规则包含 `interest`、`pleasure`、`enjoy`，因此可能删除部分普通兴趣问题，这是为了尽量降低兴趣/愉悦感问题带来的诊断提示。
5. C4 只能用于检验“显性诊断问题依赖”，不能证明模型完全不依赖访谈者侧信息。

## A10. 论文中推荐表述

推荐将 C4 表述为：

> C4 interviewer-cleaned was constructed by removing interviewer turns that directly matched depression/PHQ-style symptom probes or mental-health diagnosis/treatment-history prompts. The cleaning was rule-based, turn-level, and label-blind. It should be interpreted as a diagnostic-prompt reduction condition rather than a complete removal of interviewer-side information.
