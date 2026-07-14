# 当前论文骨架：来源相关预测信号的分层识别

本文件是投稿稿的结构索引；正文以同目录上级的 `manuscript_submission_zh.md` 为唯一权威版本。旧的“文本量不能解释”叙述不再沿用。

## 核心问题

在 DAIC-WOZ、当前特征操作化和交叉拟合模型下，完整访谈的标签相关排序信息有多少可由被试症状状态、协议路径和互动分布重建？在保留这些信息后，访谈者侧是否仍有稳定的残余增量？

## 正文四组结果

1. **单来源可预测性**：P、I、T、C5/D-P 的 AUC；排序与 Brier/校准分开报告。
2. **逐级条件增量**：P → P+Q/R → P+Q+R+D-P → 加 I；主句只说当前强基线后的残余增量。
3. **访谈者信号可恢复性**：P、Q、R、D-P 的 score-R²、区块恢复能力和 P+Q+R+D-P 后的残余标签增量；Shapley 不写成百分比，模块 15 的 `training_match_quality_limited` 必须保留。
4. **配对破坏与 Fake-D**：MPNet/BGE early concat，10 repeat×50 draw；主对比为 Real−Matched、Real−Random、Real D-P ΔAUC−Fake-D ΔAUC。

## 补充材料

PHQ 拆分、late fusion、完整 16 子集、所有指标、对称 half-min 及位置窗口、C5 retained-span alignment 细表、Brier 三种校准、完整 ledger 和 OOF 预测放入补充材料。E-DAIC 只保留描述性边界，不进入主叙事。

## 结论句

> 访谈者侧预测信号可被被试症状状态、协议路径和互动分布部分重建；真实配对保留额外排序信息，但完整控制后的残余贡献有限且依赖表示与融合方法。Fake-D 不支持把增量衰减归因于真实 D-P 症状内容的单一路径。

## 未完成的人工作业

D-P 盲法内容效度包已经生成 30 人名单和 reviewer A/B 空白表。第二名复核者和独立编码尚未回填，因此当前稿件不报告域级一致率、Cohen’s κ 或人工共识 precision/recall/F1。
