# DAIC-WOZ 与 eDAIC 去重合并计划

- 期望 DAIC-WOZ participant 数: 189
- 找不到源压缩包/文件夹: 0
- 可用原始 speaker transcript: 189

## 分类体积

- `daic_only_optional_feature`: 1513 files, 117.89 GB
- `duplicate_same_name_size`: 189 files, 5.39 GB
- `name_conflict_different_size`: 189 files, 3.07 MB

## 建议

不要整包解压到 F:。音频大概率已在 eDAIC 中重复，原始 DAIC-WOZ 的 CLNF/COVAREP/FORMANT 等特征虽不是同名重复，但体积很大，而且当前研究首先需要的是带 `speaker` 的 `*_TRANSCRIPT.csv`。

最安全做法：只抽取 `*_TRANSCRIPT.csv` 到 `F:\数据库\E-daic\DAIC-WOZ_original_transcripts`，不要放进每个 `*_P.tar\*_P` 目录，因为 Windows 下 `*_TRANSCRIPT.csv` 会和 eDAIC 的 `*_Transcript.csv` 发生同名冲突。
