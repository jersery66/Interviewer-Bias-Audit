## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-11
- Verification Status: VERIFIED
- Version Label: validation_v2_methodological_downgrade

# E-DAIC 描述性扩展分析审计报告

## 总体结论

**结论：可作为描述性补充分享，但不能作为机制验证或模型泛化证据（Share as descriptive supplement / CAUTION）。**

模块 14 在修复分块语义重复、错误分析输出契约、路径可移植性和公开隐私边界后，能够支持两项有限结论：采用 DAIC-WOZ 固定切点后，E-DAIC 新增可分析样本中仍可观察到自评标签与访谈症状证据错位；DAIC-WOZ 训练模型在该小样本上的迁移 AUC 可作带不确定区间的描述性报告。

本模块**不能独立验证“错位导致模型错误集中”**。错位分组由 `coverage_breadth` / `evidence_density` 定义，base 模型直接使用这两项特征，count / presence 模型还使用构成它们的域级特征。因此分组规则与模型输入存在定义—特征重叠（incorporation bias）；错位组较高错分率及其 Fisher p 值在结构上并非独立检验。三个迁移 AUC 的 95% CI 均跨过 0.5，也不能据此声称模型泛化性能得到验证。

模块使用编号 14 而非原方案中的 13，是因为 `13_mismatch_error_attribution/` 已存在；该调整避免覆盖已有模块，不改变研究设计。

## 方案逐项验收

| 原方案要求 | 验收结果 | 证据或说明 |
|---|---|---|
| DAIC-WOZ 为主分析，E-DAIC 仅作描述性补充 | 通过（降级后） | summary 明确限定为固定错位定义的描述性可迁移性检查 |
| 只纳入 E-DAIC 新增且公开 PHQ-8 标签者 | 通过 | 标签候选 30 人，ID 唯一；与 DAIC-WOZ 189 人 ID 交集为 0 |
| 缺失、空或格式异常 transcript 排除 | 通过 | 22 人纳入；8 人因 0 字节 transcript 排除；代码增加格式与 Text 字段校验 |
| 公开样本清单完整 | 通过 | 8 个指定字段齐全；路径改为相对 E-DAIC 根目录的可移植路径 |
| 十域 count/presence、coverage、density | 通过 | `edaic_evidence_features.csv` 含 10 域 count、10 域 presence 和两项负荷指标 |
| 抽取 prompt、模型、温度与 schema 一致 | 通过并披露偏差 | gpt-5.5、prompt v3、temperature=0；长文按 3500 字符、重叠 200 字符分块，summary 与 manifest 已披露 |
| 固定切点为 coverage >=5、density >=11 | 通过 | 主四象限使用固定切点；样本内中位数仅作敏感性描述 |
| DAIC-WOZ 142 人训练，E-DAIC 只测试 | 通过 | 训练与测试 ID 交集为 0；Scaler 仅在 DAIC-WOZ fit；E-DAIC 未用于调参或 CV |
| base/count/pres 三模型 | 通过 | 参数与方案一致；预测阈值固定为 0.5 |
| 错位、一致及四象限错误明细 | 通过，但仅作描述 | 输出 42 行：3 模型 x 2 证据定义 x 7 分组；每行标记 `independent_test=0` |
| Fisher 精确检验 | 降级为补充输出 | 保留完整 p 值用于审计；note 明确定义—特征重叠、非独立检验及 6 次比较未经校正 |
| DAIC-WOZ 与 E-DAIC 对照表 | 通过（修订后） | 删除“方向是否一致”；改为并列点估计、95% CI、样本量和限制说明 |
| manifest 与 SHA-256 清单 | 通过（增强后） | 记录输入路径与哈希、142/22 ID、参数、运行环境、脚本哈希和隐私边界 |
| 公开产物不含访谈原文 | 通过（修复后） | 公开 CSV 无 text/exact_quote 字段；含原文的输入、span、checkpoint 均被 `.gitignore` 排除 |

## 数据与计算核验

### 样本和数据质量

- E-DAIC-only 标签：30 行、30 个唯一 ID，PHQ-8 范围 0-22，无缺失。
- DAIC-WOZ-only 标签：189 行；与 E-DAIC-only 的 ID 交集为 0。
- 纳入 22/30（73.3%），排除 8/30（26.7%），排除者均为 0 字节 transcript。
- 纳入者 PHQ-8 >=10 为 6/22；排除者为 2/8。两组阳性比例 Fisher p=1.0，但不能据此排除其他选择偏倚。
- 22 份可用 transcript 均包含 `Start_Time`、`End_Time`、`Text`、`Confidence`，共 2470 行；无空文本行、无重复行。participant 660 为 199 行。
- `edaic_turns_frozen.csv` 的 `(participant_id, turn_id)` 无重复，且不发布文本内容。

### 分块证据去重修复

原合并逻辑按整行去重，分块局部的 `source_order` 不同会让同一原句逃过去重。重建 49 个已完成块后，合并前 265 条，按 `(participant_id, domain, polarity, exact_quote)` 语义去重 3 条，最终为 262 条。

去重使 participant 680 和 684 的 evidence density 各下降 1；固定切点和四象限人数不变。base AUC 从 0.6979 调整为 0.6875，presence AUC 从 0.5938 调整为 0.5833。该变化不影响错位比例的描述性结果。

### 独立复算的迁移指标

| 模型 | AUC | 分层 bootstrap 95% CI | PR-AUC | Accuracy | Sensitivity | Specificity | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.6875 | 0.3750-0.9479 | 0.5760 | 0.5909 | 0.8333 | 0.5000 | 5/8/1/8 |
| count | 0.5833 | 0.2917-0.8542 | 0.3804 | 0.5909 | 0.6667 | 0.5625 | 4/7/2/9 |
| pres | 0.5833 | 0.3229-0.8438 | 0.3435 | 0.5909 | 0.8333 | 0.5000 | 5/8/1/8 |

三种 AUC 的区间均跨过 0.5，因此只能并列报告点估计、区间和样本量；不能据点估计大于 0.5 判断“方向一致”，也不能声称稳定判别或泛化性能得到验证。

### 错分率输出的非独立性

公开补充 CSV 保留 3 个模型、2 种错位定义下的组别错分率、精确二项 95% CI 和 Fisher p 值，以便复核原始计算。但这些结果不进入主要结论：错位定义和模型预测共同使用症状证据变量，因而组间差异受到定义—特征重叠的结构性影响。Fisher 检验只能回答按该非独立规则形成的两组错分率是否不同，不能回答错位是否是模型错误的独立来源；小 p 值不会消除这一问题。

若要检验错误是否独立集中于错位样本，需要在 DAIC-WOZ 上预先训练一个不使用 `coverage_breadth`、`evidence_density`、domain count 或 domain presence 的模型，再迁移到 E-DAIC 后按固定错位定义比较错误。考虑到当前 E-DAIC 可分析样本仅 n=22、阳性 6 人，本审计不建议把追加复杂模型作为本稿的必要分析。

## 统计谬误扫描

覆盖：**11/11**。

| 类型 | 评级 | 审查结果 |
|---|---|---|
| Definition-feature overlap / incorporation bias | **HIGH** | 错位分组变量同时进入全部迁移模型；错分率差异不是独立机制检验，已从主结论中移除 |
| Simpson's paradox | NOTE | 公开结果没有性别、站点等可分层变量，无法完整排查方向反转；不据此作分层外推 |
| Ecological fallacy | 无异常 | 分析和结论均以 participant 为单位 |
| Berkson's paradox | CAUTION | 仅分析有标签且 transcript 可用者；8/30 被排除，可能存在选择机制 |
| Collider bias | 无异常 | 迁移模型未加入由标签和文本共同导致的后验控制变量 |
| Base-rate neglect | 无异常 | 明确报告 6/22 阳性比例，并同时报告 PR-AUC、敏感度和特异度 |
| Regression to the mean | 不适用 | 无前后测或按极端值选组后的变化分析 |
| Survivorship bias | CAUTION | transcript 可用率 73.3%；阳性比例相近不能排除文本特征上的选择差异 |
| Look-elsewhere effect | CAUTION | 3 模型 x 2 定义共 6 次 Fisher 检验；p 值仅保留在补充输出，不进入主结论 |
| Garden of forking paths | NOTE | 模型与固定阈值来自既定方案；API 分块是运行期修复，已完整披露并固定参数 |
| Correlation is not causation | 已处理 | summary 只报告描述性错位比例和迁移点估计，并明确不验证错位与错误之间的关系 |
| Reverse causality | 不适用 | 未主张自评与文本证据之间的因果方向 |

## 可复现性

- DAIC-WOZ 训练输入的 count、presence 与 frozen ID 集完全一致，均为 142 人。
- 10 x 5 split 无 train/test 交叉；每位参与者恰好获得 10 次测试预测。
- E-DAIC 22 个测试 ID 与 DAIC-WOZ 142 个训练 ID 交集为 0。
- 固定私有 spans 后，连续运行两次，10 个核心公开文件的 SHA-256 完全一致。
- 公开指标由预测表独立复算，所有 AUC、bootstrap CI、PR-AUC、分类指标、混淆矩阵、组别错分率和 Fisher p 与输出一致。
- C5 抽取依赖外部 gpt-5.5 API，属于随机且环境依赖的步骤，不能要求逐字节重现。其输入与含原句的 span/checkpoint 属于私有中间件，不上传 GitHub；`run_manifest.json` 保存私有 spans 的哈希以支持本地审计。

**复现判定：下游统计 REPRODUCIBLE；外部 API 抽取 N/A；整体 PARTIALLY REPRODUCIBLE。**

## 论文使用边界

必须同时保留以下限制：

1. E-DAIC 是 DAIC-WOZ 的扩展数据资源；本次新增子集虽与训练 ID 无重叠，但不是完全独立的外部数据库。
2. 实际可分析样本为 n=22，而不是候选 30；PHQ-8 阳性仅 6 人。
3. E-DAIC transcript 无 speaker 标记，C5 输入为全对话，可能包含访谈员话语干扰。
4. 固定切点下的 9/22（40.9%）只支持错位现象的描述性复现，不等于机制验证。
5. AUC 区间很宽且跨过 0.5；点估计不能写成“方向一致”或泛化性能证据。
6. 错位定义与全部迁移模型输入存在定义—特征重叠；错分率和 Fisher p 不能验证错位是错误来源。
7. PHQ-8 是自评症状量表，不是临床诊断。

论文中不应把模块 14 放入标题或摘要核心结果；适合放在结果末尾一小段或补充材料。文章的核心证据仍来自 DAIC-WOZ 主分析。

## GitHub 发布边界

公开提交包括脚本、测试、结构化 CSV、summary、manifest、SHA-256 清单和本报告。以下内容明确不提交：原始 transcript、抽取输入 JSONL、含 `exact_quote` 的 spans、分块 checkpoint、失败日志、API key 与本地调试目录。
