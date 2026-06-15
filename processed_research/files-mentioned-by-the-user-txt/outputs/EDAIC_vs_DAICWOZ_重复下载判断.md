# eDAIC 是否已包含 DAIC-WOZ：重复下载判断

## 结论

**不建议为了当前研究盲目重复下载完整 DAIC-WOZ 85GB。**

本地已下载的 eDAIC 已经包含 DAIC-WOZ 的 189 个 WoZ 会话 ID，也包含 PHQ-8 标签。因此，如果只是做“抑郁二分类”“Participant-only 文本分类”“症状证据抽取”，eDAIC 足够先跑。

但是，eDAIC 不能完全替代原始 DAIC-WOZ 包，尤其不能直接满足“访谈者偏差”这篇文章的全部设计。原因是：本地 eDAIC 的 `XXX_Transcript.csv` 只有 `Start_Time, End_Time, Text, Confidence` 四列，没有 `speaker` 列，无法可靠区分 Ellie 提问和 Participant 回答。

所以：

- 可以先不下完整 DAIC-WOZ 大包。
- 如果论文必须做 `Full dialogue / Participant-only / Interviewer-only / Interviewer-cleaned` 五条件比较，仍需要原始 DAIC-WOZ 的 speaker-separated transcript，或找到可靠的官方/可信处理版本。
- 最小补充下载优先级不是 85GB 全量包，而是先补 DAIC-WOZ 的文档和 split 小文件。

## 官方依据

USC 官网说明 DAIC-WOZ 包含 189 个 WoZ 访谈会话，每个会话包括 transcript、participant audio 和 facial features。

E-DAIC manual 说明 eDAIC 是 WOZ-DAIC 的 extended version，ID `[300,492]` 是 WoZ-controlled agent，会话 `[600,718]` 是 AI-controlled agent。

DAIC-WOZ 官方文档说明原始 DAIC-WOZ 包含 sessions 300-492，排除 342、394、398、460，共 189 个会话；并且 session 文件中包含 `XXX_TRANSCRIPT.csv`。

## 本地核查结果

本地路径：

`F:\数据库\E-daic`

核查结果：

- DAIC-WOZ 预期会话：189 个
- eDAIC archive 总数：275 个，ID 范围 300-718
- eDAIC extracted 总数：275 个，ID 范围 300-718
- DAIC-WOZ 189 个 ID 在 eDAIC archive 中缺失：0 个
- DAIC-WOZ 189 个 ID 在 eDAIC extracted 目录中缺失：0 个
- eDAIC 额外 AI-controlled ID：86 个
- `Detailed_PHQ8_Labels.csv` 覆盖 DAIC-WOZ 189 个 ID：全部覆盖

eDAIC split 情况：

- `train_split.csv`: 163 个，其中 DAIC range 143 个，AI 600+ 为 20 个
- `dev_split.csv`: 56 个，其中 DAIC range 46 个，AI 600+ 为 10 个
- `test_split.csv`: 56 个，其中 DAIC range 0 个，AI 600+ 为 56 个

这说明 eDAIC 的 train/dev/test 不是原始 DAIC-WOZ 的官方 AVEC2017 split；它把 300-492 的 WoZ 会话放在 train/dev，test 是 600+ 的 AI-controlled 会话。

## 关键限制

本地 eDAIC transcript 示例：

`F:\数据库\E-daic\data\data\300_P.tar\300_P\300_Transcript.csv`

列名是：

`Start_Time, End_Time, Text, Confidence`

没有：

- `speaker`
- `Ellie`
- `Participant`

这意味着 eDAIC transcript 不能直接构造：

- `Participant-only`
- `Interviewer-only`
- `Interviewer-cleaned`

虽然可以看到少数行里出现 Ellie 话语或诊断性问题文本，但没有可靠 speaker 标注，无法作为访谈者偏差研究的正式输入条件。

## 对当前研究的建议

### 方案 A：先不下载 DAIC-WOZ 大包

适合目标：

- 快速跑一个 eDAIC 版预实验
- 做 Participant-like text 输入
- 做症状证据抽取
- 训练传统文本基线和 LLM 零样本分类

限制：

- 不能严谨回答“访谈者问题是否造成模型偏差”
- 不能严格复现 Burdisso 等关于 Ellie prompts 的输入条件

### 方案 B：只补小文件，不补 85GB 大包

优先补：

- DAICWOZDepression_Documentation_AVEC2017.pdf
- train/dev/test split CSV
- documents.zip
- util.zip

用途：

- 写方法学说明
- 对齐官方 DAIC-WOZ split
- 确认 transcript 格式和标注手册

### 方案 C：只在必要时下载 DAIC-WOZ participant zip

如果确认必须做访谈者偏差五条件实验，则需要原始 DAIC-WOZ 的 `XXX_TRANSCRIPT.csv`。这时再下载 participant zip。

可以优先只下载 train/dev 对应 participant zip，而不是一次下载全部 189 个：

- 如果只在 train/dev 上做实验，可以先下载 train/dev ID。
- 如果需要 test 但没有 test PHQ 标签，则 test 不一定需要下载。

## 最终判断

**eDAIC 包含 DAIC-WOZ 的会话和标签，但不等于原始 DAIC-WOZ 包的完整可替代品。**

当前最理性的做法：

1. 先别重复下载 85GB。
2. 先用 eDAIC 跑 participant/evidence 方向预实验。
3. 如果论文核心仍保留“访谈者偏差”，再补原始 DAIC-WOZ speaker-separated transcripts。
