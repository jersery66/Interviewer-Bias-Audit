# DAIC-WOZ 研究数据核验报告

- 生成时间: 2026-06-12T21:30:12
- 研究主数据集: DAIC-WOZ original speaker-separated transcripts
- 输出目录: `F:\数据库\DAIC-WOZ\processed_research`
- 样本数: 189
- split 分布: {'dev': 46, 'train': 143}
- PHQ 二分类分布: {'0': 147, '1': 42}
- turn 行数: 47400
- speaker 分布: {'interviewer': 14999, 'participant': 32401}

## 与最初研究方案的对应关系

- C1 `full_dialogue`: 已生成，见 `conditions_long.csv`。
- C2 `participant_only`: 已生成，见 `conditions_long.csv`。
- C3 `interviewer_only`: 已生成，见 `conditions_long.csv`；451/458/480 没有 interviewer turn，需排除或单独报告。
- C4 `interviewer_cleaned`: 已按最初代码骨架的关键词规则生成，见 `conditions_long.csv` 和 `transcript_stats.csv`。
- C5 `symptom_evidence_only`: 还不能直接生成，因为需要先跑 LLM 症状证据抽取；输入文件已准备为 `participant_for_evidence_extraction.jsonl`。

## 当前文件是否研究需要

- `participant_index.csv`: 需要。样本、标签、split 与 transcript 统计总表。
- `labels_prepared.csv`: 需要。传统模型和 LLM 评估用标签。
- `split.csv`: 需要。固定 train/dev 划分。
- `turns.csv`: 需要。所有输入条件和后续证据定位的基础。
- `conditions_long.csv`: 需要。C1-C4 的标准长表，下一步 baseline 直接读这个。
- `transcript_stats.csv`: 需要。论文表 1/表 2 和诊断性问题数量统计。
- `participant_for_evidence_extraction.jsonl`: 需要。生成 C5 symptom evidence-only 的 LLM 输入。
- `quality_flags.csv`: 需要。记录不能进入某些条件分析的样本。
- `question_answer_pairs.csv`: 辅助需要。用于访谈者问题-患者回答配对分析和案例解释。
- `condition_status.csv`: 辅助需要。说明哪些条件 ready，哪些 pending。

## 不应作为当前主研究输入的内容

- eDAIC 的 `*_Transcript.csv` 不用于本研究主条件构造，因为它没有 speaker。
- eDAIC/DAIC-WOZ 的音频和大体积 multimodal features 暂不需要，除非以后扩展到声学/视觉模型。
- `symptom_evidence_only` 不应被假造；必须等 LLM 抽取后再加入 baseline。
