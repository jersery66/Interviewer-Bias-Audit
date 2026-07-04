# Final C4 mechanism summary

生成日期：2026-06-23 12:06:19

## 数据来源与核查边界

- 主输入来源 artifact sensitivity：由 `processed_research/turns.csv` 重新构建 raw 与 artifact-cleaned 条件，并使用同一 5×5 TF-IDF + balanced Logistic Regression 流程计算。
- C4 clean 内部消融：来自 `processed_research/c4_mechanism_v2_outputs/c4_clean_baseline_ablation_metrics.csv`。
- artifact 占比：来自 `processed_research/c4_mechanism_v2_outputs/run_log_artifact_clean_baseline.json`。
- prompt artifact 示例：优先来自 `prompt_artifact_examples.csv`，缺失类型回溯 `turns.csv` 中 C4 raw artifact 话轮。当前 C4 retained turns 未观察到 cough、sniffle、pause 或 noise 的实际示例，因此没有编造这些示例。
- 重要更正：旧 `prompt_artifact_sensitivity_main_conditions.csv` 的 C3 行实际复用了非诊断访谈者话语口径，不能作为“全部访谈者话语”结果引用。

## 10 条最终论文主张

1. C4 raw 中 prompt annotation / transcription artifact 占 56.2%，因此 raw C4 首先应解释为 artifact 敏感性结果。
2. C4 raw Macro-F1=0.696，C4 clean Macro-F1=0.579，下降 0.116，说明 raw 高表现被 artifact 明显高估。
3. C4 clean 才能作为后续非显性访谈者话语机制解释的基准。
4. 旧 `prompt_artifact_sensitivity_main_conditions.csv` 中 C3 行实际复用了非诊断访谈者话语口径，不能作为“全部访谈者话语”结果引用；本轮已重新计算 C3 all interviewer。
5. C4 clean 内部消融没有显示删除单一类别造成大幅下降，提示剩余信号较弱且具有冗余性。
6. only acknowledgment feedback 的表现较高，提示确认/倾听反馈具有高密度标签相关信息。
7. 删除 acknowledgment feedback 后 C4 clean 几乎不变，因此不能写成 acknowledgment 是唯一主导机制。
8. only grey clinical probe 表现较弱，说明灰色临床探查存在一定但有限的语义信号。
9. residual 单独较弱，但删除 residual 后下降，因此 residual 不能写成纯噪声，只能解释为可能含分散弱线索或上下文互补信息。
10. 完整转录模型性能不能直接解释为被试症状语言识别能力，必须报告输入来源、artifact 敏感性和 artifact-cleaned 机制边界。

## 必须删除或替换的旧说法

- 删除：“C4 raw 代表非显性访谈者自然话语的语义信号。”
- 删除：“非显性访谈者话语 raw 的高表现说明其具有稳定临床语义。”
- 删除：“C4 的主要机制来自 other/残余话题推进。”
- 删除：“residual 是噪声。”
- 删除：“acknowledgment feedback 是唯一主导机制。”
- 删除：“C4 clean 没有信号。”
- 替换为：“C4 raw = prompt artifact 主导；C4 clean = 弱互动反馈与少量隐性临床探查信号。”

## 输出文件

- `prompt_artifact_sensitivity_all_main_conditions.csv`
- `prompt_artifact_sensitivity_all_main_conditions.md`
- `prompt_artifact_examples_for_paper.md`
- `prompt_artifact_examples_for_supplement.csv`
- `final_table_c4_clean_ablation.md`
- `final_table_c4_clean_ablation.csv`
- `replacement_2_4_method.md`
- `replacement_3_4_results.md`
- `replacement_discussion_paragraph.md`
- `replacement_abstract_conclusion.md`
- `replacement_limitations.md`
