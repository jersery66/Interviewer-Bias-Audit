# DAIC-WOZ / eDAIC 研究数据预处理报告

- 生成时间: 2026-06-12T21:17:04
- 输出目录: `F:\数据库\E-daic\processed_research`
- participant 索引行数: 189
- split 分布: {'dev': 46, 'train': 143}
- PHQ 二分类分布: {'0': 147, '1': 42}
- turn-level 行数: 47400
- speaker 归一化计数: {'interviewer': 14999, 'participant': 32401}
- QA pair 行数: 10664

## 输出文件

- `participant_index.csv`: 每个 participant 一行，合并 split、PHQ、PTSD、路径、turn 数和文本长度。
- `turns.csv`: 原始 DAIC-WOZ speaker transcript 解析后的逐轮对话表。
- `texts_full_dialogue.csv`: 带 speaker 标签的完整对话文本。
- `texts_participant_only.csv`: 只保留 Participant 话语的文本。
- `texts_interviewer_only.csv`: 只保留 Ellie/interviewer 话语的文本。
- `question_answer_pairs.csv`: 连续 Ellie 话语块 + 紧随其后的 Participant 回答块。
- `quality_flags.csv`: 逐 participant 的质量提示，例如没有 interviewer turn 的样本。
- `processing_manifest.json`: 机器可读的处理摘要和校验结果。

## 完整性检查

- 预期 DAIC-WOZ ID: 189
- 缺标签: 0
- 缺 transcript: 0
- transcript 读取异常: 0
- 质量提示行数: 6
- 质量提示分布: {'no_interviewer_turns': 3, 'no_question_answer_pairs': 3}

## 关键注意

- 这 189 个 DAIC-WOZ 样本在当前标签文件中只有 `train` 和 `dev`，没有 DAIC-WOZ test 标签。
- eDAIC 的 `test_split.csv` 是 600+ ID，不应混进 DAIC-WOZ 原始 189 人实验。
- 原始 DAIC-WOZ 的 `*_TRANSCRIPT.csv` 被单独放在 `DAIC-WOZ_original_transcripts`，避免和 eDAIC 的 `*_Transcript.csv` 在 Windows 下发生大小写同名冲突。
