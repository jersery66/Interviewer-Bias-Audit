# 半结构化抑郁访谈中来源相关预测信号的分层识别：DAIC-WOZ PHQ-8 分析

**英文拟题：** Layered identification of source-related predictive signals in semi-structured depression interviews

> **主张边界：** 本研究识别的是在 DAIC-WOZ、当前特征操作化和交叉拟合模型下的标签相关预测信息。结果不证明访谈者造成了偏差，不证明存在标签泄漏，也不证明任何文本来源具有因果意义或临床部署价值。

## 摘要

半结构化访谈的完整转录同时包含被试回答、访谈者语言、协议路径、互动分布以及由症状证据构建的衍生表示。若不分解这些来源，完整转录的预测性能不能直接解释为被试症状语言，也不能由单来源 AUC 推出访谈者文本具有独立贡献。本研究对 DAIC-WOZ 官方 train+dev 队列中的 142 名受试者进行来源审计，其中 PHQ-8≥10 者 43 名。分析主线包括四组结果：单来源可预测性、逐级条件增量、访谈者信号的统计可恢复性及残余标签增量、以及配对破坏与 Fake-D 识别敏感性。所有主要预测均在受试者级外层五折中交叉拟合；第 1 组划分在本轮投稿审计前锁定，10×5 重复划分仅用于稳定性描述。

被试语言、访谈者语言、完整转录和被试症状证据的 AUC 分别为 0.717、0.808、0.760 和 0.735。访谈者语言相对于已包含被试语言、协议结构和 D-P 症状域的强基线，其条件 ΔAUC 为 0.0028（95% CI −0.0040 至 0.0101，BH q=0.473）。模块 15 的交叉拟合结果显示，P、Q、R、D 均可在不同表示下部分重建访谈者侧预测分数，但 R 的统计恢复能力通常最大；Shapley 输出被作为重叠信息的描述性分解，不被解释为信号百分比。完整 P+Q+R+D 后的残余标签增量随 TF-IDF、MPNet 和 BGE 表示改变，未形成稳定方向；该正式结果的运行状态标记为 `training_match_quality_limited`，因此其配对部分不作为主要识别证据。

在新锁定的 10-repeat 早期拼接分析中，每个 repeat 使用 50 次 Random、50 次结构匹配 Matched 和 50 次 Fake-D。Real−Matched 的平均 ΔAUC 为 MPNet 0.0401（95% CI 0.0287–0.0509）和 BGE 0.0592（0.0509–0.0687）；Real−Random 分别为 0.0555（0.0430–0.0679）和 0.0826（0.0722–0.0936）。十个 repeat 的方向均一致，六项主要对比的 exact sign-flip p 均为 0.001953，BH q 亦为 0.001953。相反，真实 D-P ΔAUC 与 Fake-D ΔAUC 的差值为 MPNet −0.0373（−0.0514 至 −0.0237）和 BGE −0.0729（−0.0871 至 −0.0589），说明该负对照没有显示“只有真实症状内容才会吸收访谈者增量”的模式。综合结果支持一个较窄的结论：真实配对的访谈者表示保留额外排序信息，但其完整条件增量有限，且对表示、结构控制和 D-P 操作化敏感。

**关键词：** DAIC-WOZ；PHQ-8；访谈者语言；来源分解；条件增量；配对破坏；负对照

## 1 引言

DAIC-WOZ 等半结构化抑郁访谈数据被广泛用于从转录文本预测 PHQ-8 标签[1,2]。然而，完整转录并不是单一的“被试语言”通道：访谈者按照协议提问，受试者回答会改变追问路径，协议模板与问题覆盖会改变文本长度和信息分布，后续处理还可能产生症状证据表示[3–9]。因此，完整转录的 AUC 只能说明混合输入含有标签相关排序信息，不能单独说明模型识别了被试症状表达。

本文把“某一来源能否预测”与“该来源在保留其他信息后是否增加信息”分开。单来源模型回答预测充分性问题；逐级条件模型回答在当前基线下的残余增量问题；进一步的信号可恢复性分析回答哪些可观测信息块能够统计重建访谈者模型分数；配对破坏与 Fake-D 则分别检验真实配对信息和症状域负对照。研究不把这些统计关系写成访谈者偏差的因果识别。

本轮相关工作的定向检索纳入了访谈者提示、协议结构、互动建模和来源偏差控制等代表性研究；检索范围、纳入与排除标准记录在 `analysis_v2/12_submission_audit/targeted_literature_search_log.md`。本文不据此提出“首次”或“唯一”的新颖性结论。

本文的分析范围有三项明确限制。第一，D 是由被试语言衍生的十个症状域计数（D-P），不是从完整访谈重新提取的 D-All。第二，所有新增分析均为审计后提出的敏感性分析，不是预注册。第三，D-P 的人工盲法内容效度核查目前只完成了抽样合同和盲法表，尚未得到两名独立复核者的回填，因此本文不报告人工一致率、Cohen’s κ 或人工共识 F1。

## 2 方法

### 2.1 数据、队列与结局

分析使用 DAIC-WOZ 官方 train+dev 队列中具有冻结文本、结构变量和 PHQ-8 标签的 142 名受试者（阳性 43、阴性 99）。PHQ-8≥10 是本文的二元建模标签；PHQ-8 是自评症状量表，不等同于临床诊断或真实世界筛查结局[12]。官方测试集和 E-DAIC 不进入当前主分析。

主结果使用投稿审计开始前锁定的第 1 组五折受试者级划分，每名受试者在外层测试中产生一次 cross-fitted 预测。已有 10×5 重复划分用于稳定性和新增正式敏感性分析；划分锁定不构成预注册。

### 2.2 输入来源与信息块

主要来源为：

* **P：** 被试语言或被试侧来源模型得分；
* **I：** 访谈者语言；
* **T：** 完整转录；
* **D-P：** 最终保留的 1,137 条 C5 证据所生成的十个被试语言症状域计数：anhedonia/interest、appetite/weight、concentration/psychomotor、depressed mood、functioning impairment、mental-health history、protective/absent symptom、self-worth/guilt、sleep/fatigue/energy、suicide/self-harm；
* **Q：** 文本量和互动分布，包括被试/访谈者词数、轮次数、平均每轮词数和访谈时长；
* **R：** 协议路径和提示覆盖，包括临床问题数、提示覆盖、临床提示在访谈中的位置、转换次数、连续临床提示长度和冻结 prompt ID 的计数/出现变量。

D-P 的来源审计显示，1,137/1,137 个最终保留证据片段可在被试语言中追溯，未发现仅能由访谈者文本匹配的片段。该审计只说明来源边界，不等于人工内容效度验证，也不构成 D-All。

### 2.3 模型与交叉拟合

TF-IDF 模型使用固定的词项配置和带平衡类别权重的 L2 逻辑回归。MPNet 与 BGE 使用已冻结、L2 归一化的句向量缓存；新增配对/Fake-D 主分析只使用 MPNet 和 BGE 的 early concat。所有文本变换、数值标准化、C 选择和校准均限制在外层训练折；外层测试折不参与来源模型、融合模型或校准器拟合。

逐级条件模型按“被试语言 → 被试语言+协议/互动结构 → 加 D-P → 加访谈者语言”的顺序报告。条件增量以配对 OOF 预测的 ΔAUC 为主，同时保留 PR-AUC、Brier 和 log loss 的差值。概率质量使用外层训练折阳性率构建的正式无技能基线：

概率质量使用外层训练折阳性率构建的正式无技能基线，并以 Brier skill score (BSS) 报告：

\[
BSS=1-\frac{Brier_{model}}{Brier_{null,outer\ training\ fold}}.
\]

全队列阳性率 43/142 只作描述性基线。Raw、Platt 和 isotonic 在同一批外层 cross-fitted 预测上比较；isotonic 的方差不因单次 Brier 较低而被解释为“最优”。

### 2.4 访谈者信号可恢复性

模块 15 将访谈者来源模型的外层训练折隔离 logit 作为目标，使用 Ridge 在每个外层折内拟合 16 个 P/Q/R/D 子集。其输出包括 R²、MAE、RMSE、Spearman ρ 和 MSE 改善。区块 Shapley 仅表示在该表示、模型和样本中的重叠统计恢复能力，不是因果贡献，也不是信号百分比。

在残余分析中，先用 P+Q+D 预测访谈者分数，再比较加入 `residual_without_R` 与完整 P+Q+R+D；该操作只检验当前特征空间中的标签增量。模块 15 的最终目录已经完成双跑、哈希和 `final/` 晋级，但 manifest 明确标记 `training_match_quality_limited`；因此本文使用其可恢复性和残余结果作描述，不把其匹配结果写成稳定的主检验。

### 2.5 配对破坏与 Fake-D 正式敏感性

这部分是本轮锁定的正式 post-audit 敏感性分析，仅包含 MPNet、BGE、early concat、10 个既有 repeat。每个 repeat 运行：

1. **Real：** 保持 P 与 I 的真实受试者配对；
2. **Random：** 用预先固定种子的全队列无自配对置换；
3. **Matched：** 以访谈者词数、问题数、临床问题数、唯一 prompt 数、临床 prompt 覆盖和非显式访谈者轮次构成的结构距离做 Hungarian 一对一匹配，并禁止自配对；
4. **Fake-D：** 保持 P 不变，将完整 D-P 行在受试者之间做标签盲的无自配对置换。

每个 repeat 的 Random、Matched 和 Fake-D 各有 50 个 draw；相同 draw ID 与种子在两个 embedding 间共享。每个 draw 都重新完成五折建模，差值在同一受试者、同一 repeat 和同一 draw 内配对。主要只保留三个对比：Real−Matched、Real−Random、以及真实 D-P ΔAUC−Fake-D ΔAUC。先在 repeat 内对 50 个 draw 求均值，再以十个 repeat 的对比值计算 5,000 次 bootstrap 百分位 95% CI 和 exact two-sided sign-flip p；BH-FDR 作用于两个 embedding×三个对比共六行。

### 2.6 C5 来源对齐与 D-P 盲法核查

C5 的来源对齐敏感性分析只比较同一批 1,137 条最终保留证据：原始模型 quote 与人工逐字来源对齐 quote。它不能恢复 5 条重复证据、1 条无效证据或候选筛选前未保留内容，因而不称为审核前后完整比较。该分析报告 AUC、PR-AUC、Brier、配对概率变化及 35 条修订涉及的 24 名受试者；域、极性和保留状态在比较中固定。

D-P 盲法内容效度包按标签×被试词数三分位分层抽取 30 人（每格 5 人，种子 20260714）。复核者只获得去除标签、PHQ、模型结果、D-P 和访谈者文本的被试语言；每人逐域记录是否出现和逐字证据。当前仓库只包含空白 reviewer A/B 表及 owner-only 取样键；在回填前不计算一致率、Cohen’s κ、共识 precision、recall 或 F1。

## 3 结果

### 3.1 单来源可预测性：来源均含有排序信息

在锁定五折中，P、I、T 和 C5/D-P 的 ROC-AUC 分别为 0.717（95% CI 0.614–0.811）、0.808（0.729–0.879）、0.760（0.670–0.844）和 0.735（0.640–0.819）。I 的点估计高于 P，但配对区间跨过零，单来源 AUC 不足以判定访谈者更重要。P 的 raw Brier 约为 0.231；相对于外层训练折基率的 raw BSS 为 −0.092，Platt BSS 为 0.152。该结果区分了排序信息和概率质量，不能被写成“模型无信号”或临床失效。

### 3.2 逐级条件增量：强 D-P 基线后的 I 增量很小

在包含 P、协议/互动结构和 D-P 的强基线之上加入 I，ΔAUC=0.0028，95% CI −0.0040 至 0.0101，raw p=0.4731，BH q=0.473。这个估计回答的是“在当前 D-P 与结构变量已经进入模型后，原始访谈者词汇表示还能增加多少排序信息”，不是“访谈者语言整体没有价值”。协议结构和全部结构变量自身也含有标签相关信息；interaction-only 的不确定性较大，不能把结构写成因果机制。

文本预算敏感性进一步限制了来源解释。原有较长来源向较短来源截断是不对称控制；对称 half-min 预算的点估计为访谈者−被试 0.1174，但同时纳入受试者抽样和窗口随机性的联合 95% CI 为 −0.0691 至 0.2953。前、中、后固定窗口给出的来源差异明显不同。因此，文本量不能被当前结果完全排除，也不能被写成已经排除。

### 3.3 访谈者侧信号可恢复性与残余标签增量

模块 15 的 repeat 1 交叉拟合分数恢复结果如下（R² 为统计恢复指标，不是方差解释的因果比例）：

| 表示 | P | Q | R | D-P | P+Q+R+D-P |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.241 | 0.203 | 0.758 | 0.229 | 0.780 |
| MPNet | 0.166 | 0.219 | 0.290 | 0.074 | 0.297 |
| BGE | 0.131 | 0.110 | 0.390 | 0.117 | 0.424 |

三种表示中，R 的区块 Shapley 估计均为最大：TF-IDF 0.491（95% CI 0.433–0.553）、MPNet 0.144（0.015–0.255）、BGE 0.289（0.218–0.356）。D-P 的估计依表示而变：TF-IDF 0.100（0.067–0.135）、BGE 0.044（0.013–0.077），而 MPNet 为 −0.003（−0.055–0.043）。这些数值说明协议路径、数量/互动、被试分数和 D-P 在不同表示下具有重叠的统计恢复能力，不提供“信号由各块按百分比组成”的解释。

在 P+Q+R+D-P 之后测试残余标签增量，repeat 1 的 ΔAUC 为 TF-IDF −0.0242（95% CI −0.0517 至 −0.0005，p=0.0460）、MPNet −0.0216（−0.0524 至 0.0059，p=0.1400）和 BGE 0.0139（−0.0099 至 0.0390，p=0.3032）。10-repeat 描述性均值分别为 −0.0046、−0.0280 和 −0.0115，但该模块的最终 manifest 标记 `training_match_quality_limited`，所以这些数值被作为表示敏感性的描述，不被概括为跨表示确认性检验。当前证据支持“完整控制后的残余增量有限且依赖表示”，不支持“访谈者语言没有任何信息”。

### 3.4 配对破坏与 Fake-D：真实配对可保留排序差异，但 D-P 负对照不支持单一路径解释

正式 10-repeat 结果只使用 MPNet/BGE early concat。每个 embedding 的 10 个 repeat 均完成 50 个 Random、50 个 Matched 和 50 个 Fake-D draw；随机和匹配 ledger 无自配对，OOF 预测覆盖 142 名受试者。

| 表示 | 主要对比 | 10-repeat 平均差值 | 95% CI | 方向一致（正/负） | p | BH q |
|---|---|---:|---:|---:|---:|---:|
| MPNet | Real−Matched | 0.0401 | 0.0287–0.0509 | 10/0 | 0.001953 | 0.001953 |
| MPNet | Real−Random | 0.0555 | 0.0430–0.0679 | 10/0 | 0.001953 | 0.001953 |
| MPNet | Real D-P ΔAUC−Fake-D ΔAUC | −0.0373 | −0.0514–−0.0237 | 0/10 | 0.001953 | 0.001953 |
| BGE | Real−Matched | 0.0592 | 0.0509–0.0687 | 10/0 | 0.001953 | 0.001953 |
| BGE | Real−Random | 0.0826 | 0.0722–0.0936 | 10/0 | 0.001953 | 0.001953 |
| BGE | Real D-P ΔAUC−Fake-D ΔAUC | −0.0729 | −0.0871–−0.0589 | 0/10 | 0.001953 | 0.001953 |

真实配对相对于随机或结构匹配配对保留了额外的受试者级排序差异。这个结果支持“访谈者表示包含与真实配对有关的可预测排序信息”，但不识别其来源是互动语义、未观测协变量还是协议相关结构。更关键的是，真实 D-P ΔAUC 并未高于 Fake-D；两种表示下差值均为负，且十个 repeat 方向一致。因此不能把访谈者增量下降归因于 D-P 已经特异性吸收了患者症状内容。该负对照同时与结构化变量效应、标签同构、高维噪声或表示/模型交互相容。

## 4 讨论

### 4.1 主要发现与最窄可辩护结论

本文首先复现了一个容易被误读的事实：I 单独具有较高 AUC，但这只是标签相关排序能力。第二，加入 D-P、协议和互动结构后，I 的残余 ΔAUC 很小，且其方向在 TF-IDF、MPNet 和 BGE 间不完全一致。第三，访谈者侧分数可以被 P、Q、R、D-P 统计重建，其中 R 通常对应较大的恢复能力；这使“完整转录性能全部来自被试症状语言”的解释不充分，但也不允许把 R 的统计恢复写成协议造成访谈者判断。第四，真实配对相对于随机和结构匹配配对仍有额外排序差异，但 Fake-D 对比没有显示真实 D-P 特异性地吸收 I 的标签增量。

因此，本文最窄的结论是：

> 在 DAIC-WOZ、当前输入操作化和交叉拟合模型下，访谈者侧预测信号可以由被试症状状态、协议路径和互动分布部分重建；真实配对保留额外排序信息，但完整控制后的残余贡献有限，并依赖表示与融合方法。现有设计不能把该结果提升为访谈者偏差的因果识别。

### 4.2 为什么不能把 I 增量下降写成“症状内容被吸收”

D-P 是由被试语言派生、且与 PHQ-8 构念相近的结构化表示。因此加入 D-P 后 I 增量下降至少有四种解释：症状内容重叠、D-P 的特征工程优势、标签同构、以及高维模型中的噪声或竞争。真实 D-P 与 Fake-D 的正式对比没有支持“只有真实症状内容才会削弱 I”这一单一路径。更合适的写法是“在当前 D-P 表示下观察到条件衰减，但其识别解释仍受表示和负对照结果限制”。

### 4.3 文本量、窗口和校准

对称 half-min 结果的联合区间跨过零，且前、中、后窗口差异明显，说明可用上下文位置不可忽略。原始 Brier 低于训练折无技能基线并不等同于概率可用；本队列中 Platt 校准改善了 Brier skill，isotonic 的不确定性应单独报告。固定 0.50 的灵敏度问题是概率尺度和操作点问题，不能直接写成模型没有排序信号。

### 4.4 C5 的定位

C5 是临床导向的衍生表示，不是独立自然语言来源。最终保留片段的来源对齐修订只改变了逐字 quote，域、极性和保留状态固定；AUC 变化 −0.0042 的区间跨零。因此，该修订主要改善可追溯性，不能据此声称人工审核前后完整性能未变。由于 D-P 盲法人工核查尚未回填，C5 的人工内容效度与复核者一致性仍是未解决的质量问题。

## 5 局限

1. 样本仅 142 人，且标签是 PHQ-8 自评阈值；结果不等同于诊断或临床部署性能。
2. 所有结果来自单一 DAIC-WOZ 队列；当前阶段不做 E-DAIC 外部验证，也不把已有小规模扩展写成复制或泛化验证。
3. D-P 是被试语言衍生的十域计数，缺少合格的完整访谈 D-All；因此“控制症状域”不能理解为控制了所有访谈症状内容。
4. 10-repeat Fake-D 和配对分析使用当前特征操作化与 early concat，不能识别因果机制；Fake-D 的负结果反而限制了症状吸收解释。
5. 模块 15 最终 manifest 的 `training_match_quality_limited` 标记限制了其匹配配对结果的主要解释；可恢复性和残余结果仍是模型/表示条件下的统计量。
6. D-P 盲法核查目前只有空白双评审表和 owner-only 键，尚未有第二名复核者的独立编码；在此之前不报告 Cohen’s κ、共识 F1 或人工准确性。
7. 对称文本预算和位置敏感性仍不能区分语义、信息密度和窗口位置；原始不对称控制保留作补充，二者回答的问题不同。
8. 本轮分析与合同是在审计后锁定，不是预注册；所有新增结果必须与主分析分开标注。

## 6 结论

在 DAIC-WOZ 的受试者级交叉拟合中，被试语言、访谈者语言、协议结构、互动分布和 D-P 均可携带标签相关信息。访谈者语言的单来源 AUC 不能直接解释为独立价值；在加入 P、Q、R 和 D-P 后，其残余增量小且随表示变化。真实配对相对于随机/结构匹配配对仍保留额外排序差异，但真实 D-P 与 Fake-D 的正式对比不支持把增量下降归因于症状内容被特异性吸收。更稳妥的结论是：完整访谈中的预测信号由多个相互重叠的来源共同构成，来源、信息预算、协议路径和表示方式必须在解释模型性能时同时报告。

## 数据与代码可用性

代码、测试、冻结输入的哈希、交叉拟合派生结果和正式敏感性输出保存在 `analysis_v2/`。配对/Fake-D 正式结果入口为 `analysis_v2/10_source_importance/08_identification_sensitivity/run_formal_10x/`；模块 15 结果入口为 `analysis_v2/15_interviewer_signal_explanation/final/`；D-P 盲法包入口为 `analysis_v2/04_c5_controls/blind_quality_audit/packet_1/`。公开 CI 只验证 code/test reproducibility、合成 fixture、schema/路径检查、图表可执行性以及已提交派生结果的哈希一致性。没有 DAIC-WOZ 受限原始数据和私有 C5 审计源时，不声称干净克隆可以从原始数据完整复现全部结果。

## 伦理声明

本研究是对既有去标识化研究数据的二次分析。原始数据伦理审批、知情同意与共享限制以数据提供方文件为准。本文不作临床诊断、治疗决策或部署用途声明，模型输出不得替代专业评估。

## 作者贡献、利益冲突与经费

作者贡献、利益冲突和经费信息应在投稿系统中按实际情况填写；本分析文件不据未提供的信息代填作者身份或资助来源。

## AI 工具使用声明

代码审计、结构化整理和文字重写使用了 AI 辅助工具；分析定义、数据边界、结果核对和最终学术判断由作者负责。AI 工具未被授权访问未提交的受限原始数据或私有 C5 原句源。

## 参考文献

1. Gratch J, Artstein R, Lucas GM, et al. The Distress Analysis Interview Corpus of human and computer interviews. *LREC 2014*. 2014:3123–3128. https://aclanthology.org/L14-1421/
2. Valstar M, Gratch J, Schuller B, et al. AVEC 2016: Depression, Mood, and Emotion Recognition Workshop and Challenge. *AVEC 2016*. 2016:3–10. doi:10.1145/2988257.2988258
3. López-Otero P, Docío-Fernández L, García-Mateo C. Depression Detection Using Automatic Transcriptions of De-Identified Speech. *Interspeech 2017*. 2017:3157–3161. doi:10.21437/Interspeech.2017-1201
4. Mallol-Ragolta A, Zhao Z, Stappen L, Cummins N, Schuller BW. A Hierarchical Attention Network-Based Approach for Depression Detection from Transcribed Clinical Interviews. *Interspeech 2019*. 2019:221–225. doi:10.21437/Interspeech.2019-2036
5. Rinaldi A, Fox Tree J, Chaturvedi S. Predicting Depression in Screening Interviews from Latent Categorization of Interview Prompts. *ACL 2020*. 2020:7–18. doi:10.18653/v1/2020.acl-main.2
6. Burdisso S, Reyes-Ramírez E, Villatoro-Tello E, et al. DAIC-WOZ: On the Validity of Using the Therapist's Prompts in Automatic Depression Detection from Clinical Interviews. *ClinicalNLP 2024*. 2024:82–90. doi:10.18653/v1/2024.clinicalnlp-1.8
7. Agarwal N, Milintsevich K, Metivier L, et al. Analyzing Symptom-based Depression Level Estimation through the Prism of Psychiatric Expertise. *LREC-COLING 2024*. 2024:974–983. https://aclanthology.org/2024.lrec-main.87/
8. Zhao X, Lyu Y, Wang D, Tang B. Predicting Depression in Screening Interviews from Interactive Multi-Theme Collaboration. *Findings of ACL 2025*. 2025:23025–23035. doi:10.18653/v1/2025.findings-acl.1181
9. Zhang E, Poellabauer C. Mitigating Interviewer Bias in Multimodal Depression Detection: An Approach with Adversarial Learning and Contextual Positional Encoding. *Findings of EMNLP 2025*. 2025:12169–12188. doi:10.18653/v1/2025.findings-emnlp.650
10. Lee JJ, Han J, Woo CW. Interpretable depression assessment using a large language model. *PLOS Digit Health*. 2026;5(2):e0001205. doi:10.1371/journal.pdig.0001205
11. Schmidt F, Ravan S, Vlassov V. Probabilistic Depression Detection from Textual Time Series. *Findings of ACL 2026*. 2026:32574–32589. doi:10.18653/v1/2026.findings-acl.1630
12. Kroenke K, Strine TW, Spitzer RL, Williams JBW, Berry JT, Mokdad AH. The PHQ-8 as a measure of current depression in the general population. *J Affect Disord*. 2009;114(1–3):163–173. doi:10.1016/j.jad.2008.06.026
13. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B*. 1995;57(1):289–300. doi:10.1111/j.2517-6161.1995.tb02031.x
