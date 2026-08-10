# 参数选择依据与溯源

## 1. 文档目的

本文档专门回答“参数为什么这样选”。核心原则是区分四种来源：

- **A. 数据集或已发表任务定义直接决定**：例如 PHQ-8 二分类阈值、说话者拆分和症状层级分析。
- **B. 模型规范或折内选择决定**：例如 Logistic Regression 的候选 `C`、训练折内标准化和校准。
- **C. 表示稳健性与资源约束决定**：例如 MPNet/BGE、chunk、pooling 和 L2 normalization。
- **D. 推断精度、计算预算与冻结协议决定**：例如 repeats、draws、bootstrap、permutation 和随机种子。

不是每个数值都必须由某篇论文逐字规定。对于没有直接文献常数的参数，合格依据是：在看正式结果前冻结、与研究问题匹配、外层测试折不参与选择、计算上可行、能够量化所需不确定性，并在 manifest 中可复核。

本文档记录实际执行配置，不修改任何结果，也不把事后锁定写成预注册。

## 2. 参数依据总表

### 2.1 队列、标签与输入来源

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 冻结与溯源 | 解释边界 |
|---|---|---|---|---|---|
| 主队列 | 官方 train+development，共 142 人 | A | 这些参与者具有研究所需标签并构成现有正式分析共同队列；保持全部模型同队列可做参与者级配对比较。 | `analysis_v2/00_plan/analysis_plan_v2.md`；`analysis_v2/12_submission_audit/run_manifest.json` | 不是外部验证；官方 test 不构成本研究独立外部队列。 |
| 二分类标签 | `PHQ-8 >= 10` | A | 继承 DAIC-WOZ 相关文献中的标准二分类操作化；Milintsevich 等明确使用总分 10 作为二分类阈值。 | [Milintsevich et al., 2023](https://doi.org/10.1186/s40708-023-00185-9)；冻结输入标签列 | 标签是自评筛查操作化，不是临床诊断。 |
| 输入来源 | 完整转录、参与者文本、访谈者文本、去显性提示访谈者文本、参与者症状证据 | A/B | Burdisso 直接区分参与者回答与访谈者提示；Agarwal 将 patient/therapist 拆为不同视角。 | `analysis_v2/00_plan/analysis_plan_v2.md`；[Burdisso et al., 2024](https://doi.org/10.18653/v1/2024.clinicalnlp-1.8)；[Agarwal et al., 2024a](https://doi.org/10.18653/v1/2024.clpsych-1.9) | 来源定义不是按结果调整的文本选择器。 |
| `D-P` 症状域 | 10 个冻结域计数 | A/B | 症状级建模有直接文献先例，但本研究需要覆盖当前症状、功能、历史、保护/否认及自伤等研究问题，因此不是机械复制 PHQ-8 八项。 | `analysis_v2/03_embedding_robustness/embedding_incremental_analysis_plan.md`；`analysis_v2/15_interviewer_signal_explanation/interviewer_signal_explanation_plan.md` | 不能称为 PHQ-8 逐项真值、D-All 或独立临床测量。 |

### 2.2 交叉验证、调参与数据泄漏控制

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 冻结与溯源 | 解释边界 |
|---|---|---|---|---|---|
| 外层划分 | 参与者级分层 5-fold | B | 在 142 人小样本中兼顾训练量和测试覆盖；同一参与者只进入一个外层测试折，所有来源共享划分以支持成对比较。 | `analysis_v2/00_splits/repeated_5fold_splits_10x5.csv`；manifest 记录哈希 | fold 不是独立样本；不得用折均值替代参与者级 OOF 推断。 |
| 重复次数 | 10 repeats × 5 folds | D | 10 组预先生成的划分用于检查分割敏感性；正式配对分析以 10 个 repeat 对比均值作为推断单位。 | split 文件与正式配对 manifest | 10 repeats 不等于 10 个独立队列；不把 50 folds 当作独立样本。 |
| 分析锁定主划分 | repeat 1 的 5 folds | D | 投稿审计前锁定一个主划分，避免在重复划分中挑选有利结果；其余 repeats 用于稳定性或指定的正式重复级推断。 | `analysis_v2/12_submission_audit/model_configuration.md`；`run_manifest.json` | 是 analysis-locked，不是原始预注册。 |
| 内层划分 | 3-fold stratified CV | B | 在外层训练样本有限的情况下为超参数、校准和阈值选择保留可用验证量；测试折始终不可见。 | embedding incremental plan 与 manifests | 不是对整个 142 人先调参再交叉验证。 |
| Dense Logistic `C` 候选 | `[0.01, 0.1, 1.0, 10.0]` | B | 在对数尺度覆盖从强到弱正则化的紧凑候选空间；最高内层平均 ROC-AUC 胜出，同分选更小 `C`。 | `reanalysis_v2/embedding_incremental.py`；embedding manifests | 候选空间是研究设计，不是文献规定的唯一正确网格。 |
| TF-IDF Logistic `C` | `1.0` 固定 | B | 继承分析锁定的原模型配置，避免投稿审计后重新搜索超参数改变主结果。 | `analysis_v2/12_submission_audit/model_configuration.md` | 不能声称 `C=1` 在该队列最优；dense 增量模型另行折内选 `C`。 |
| 预处理拟合范围 | vocabulary、TF-IDF 权重、标准化、校准、阈值均仅用外层训练数据 | B | 防止外层测试参与者影响特征空间、概率尺度或决策规则。 | `model_configuration.md`；embedding incremental plan | 这是无泄漏要求，不是可选敏感性设置。 |

### 2.3 TF-IDF 与分类器

| 参数 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| lowercase | `True` | B | 合并英文大小写变体，降低小样本词表稀疏性。 | `reanalysis_v2/modeling.py`；`model_configuration.md` | 不是临床语义假设。 |
| strip accents | `unicode` | B | 统一 Unicode 字符变体，减少同词表面差异。 | 同上 | 不表示语音或方言被建模。 |
| word n-grams | `(1, 2)` | B | unigram 表示词项，bigram 保留有限局部搭配；在小样本中避免更高阶 n-gram 的过度稀疏。 | 同上 | 不能据此声称捕获长程语义。 |
| `min_df` | `2` | B | 去除只在一个文档出现的极稀有项，降低单个参与者特征化风险。 | 同上 | 不代表罕见临床词不重要。 |
| `max_features` | `50,000` | B | 对词表规模设可复现上限，同时保留足够 unigram/bigram；该值在审计前冻结。 | 同上 | 不是经外部测试优化的最优维数。 |
| sublinear TF | `True` | B | 对重复词频使用对数缩放，避免长访谈中的高频词线性支配。 | 同上 | 不能消除全部文本长度影响，因此另有文本预算控制。 |
| 分类器 | L2 Logistic Regression | B | 线性、可重复、适合高维稀疏或冻结 dense 表示；把研究重点放在输入来源而非复杂模型架构。 | `reanalysis_v2/modeling.py`；embedding plan | 结论是该模型族下的来源审计，不是所有模型的定理。 |
| class weight | `balanced` | B | 队列标签不平衡，训练时按类别频率加权，避免多数类支配损失。 | 同上 | 不改变真实患病率，也不等于概率已校准。 |
| solver | `liblinear` | B | 与 L2 Logistic 和小样本设置兼容，并保持原分析配置一致。 | 同上 | 不声称相对其他 solver 性能最优。 |
| `max_iter` | `2,000` | B | 提供充分迭代上限以降低未收敛风险；不是效果调优参数。 | 同上 | 迭代上限不代表实际每次都运行 2,000 步。 |

### 2.4 Dense 表示与长文本聚合

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 冻结与溯源 | 解释边界 |
|---|---|---|---|---|---|
| Dense model 1 | `sentence-transformers/all-mpnet-base-v2`，768 维，revision `e8c3b32...a19130` | C | 通用英文句向量；Agarwal 的 DAIC-WOZ 图模型也使用该模型生成句级表示，提供直接任务先例。 | [Agarwal et al., 2024c](https://doi.org/10.1186/s40708-024-00227-w)；模型 manifest | 先例支持“可用作表示”，不证明本研究分类器或结果。 |
| Dense model 2 | `BAAI/bge-large-en-v1.5`，1024 维，revision `d4aa690...e09` | C | 作为性质和维数不同的第二 dense 表示检查稳健性；在正式执行前因 6GB GPU/CPU 环境无法普通运行 7B 候选而替换，并非看结果后选择。 | `analysis_v2/00_plan/changelog.md`；BGE manifest | 不称为最佳模型；两种已执行 dense 表示必须同时报告。 |
| 设备 | CPU | C | 当前环境中模型编码可重复且避免无法满足的 7B GPU 内存要求；运行时间记录在 manifest。 | 两个 embedding manifests | 设备影响资源与时间，不应被解释为模型效果来源。 |
| chunk 长度 | 200 words | C | 避免将整段长访谈直接截断，同时保留局部上下文；在所有输入条件和两种模型中统一。 | embedding manifests | 是工程操作化，不是发表文献规定的唯一最优窗口。 |
| chunk overlap | 50 words | C | 使跨 chunk 边界的局部内容有重复覆盖，降低硬切分边界损失。 | embedding manifests | 不等于保留完整长程对话依赖。 |
| chunk pooling | 归一化 chunk embedding 的非加权均值 | C | 把可变数量 chunk 聚合为固定参与者级向量，并避免长文档仅因 chunk 数量获得更大向量范数。 | embedding manifests | 均值可能平滑局部稀有证据；结论依赖此聚合操作化。 |
| 文档归一化 | L2 normalization | C | 统一参与者级向量尺度，使线性分类器主要利用方向而非总范数。 | embedding manifests 与 embedding plan | 不消除词量或位置差异，因此另有专门控制。 |
| 空文档策略 | 全零向量 | C | 提供确定、可审计且不注入伪文本的缺失表示。 | embedding manifests | 全零只表示该输入条件无文本，不是症状阴性。 |

### 2.5 融合、控制块与可恢复性

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| 条件增量层级 | L1: P→P+I；L2: P+R→P+R+I；L3: P+D→P+D+I；L4: P+R+D→P+R+D+I | C | 逐层把协议和症状替代解释加入，直接区分充分性与剩余增量。 | embedding incremental plan | 层级是条件比较，不是因果调整序列。 |
| late fusion | 第一层严格交叉拟合 logits；第二层训练折内标准化 | B/C | 避免第二层看到由同一行训练得到的源分数，并以低维分数融合来源。 | embedding incremental plan | late fusion 的衰减不能写成信号百分比。 |
| early concat | 冻结 embeddings 直接拼接；只标准化 R/D 数值列 | B/C | 检查融合方法是否决定结果，同时保留已 L2 归一化的 dense 向量尺度。 | embedding incremental plan | 与 late fusion 差异说明模型操作化敏感，不自动说明机制。 |
| `Q` | 9 个数量/互动变量 | C | 控制词量、时长、话轮与互动分布等非词汇替代解释。 | interviewer signal explanation plan | Q 不是问答语义或互动质量。 |
| `R` 基础块 | 5 个冻结协议计数 | B/C | 对提示数量、临床问题、提示覆盖及非显性话轮进行最低限度协议控制。 | structural feature dictionary；embedding plan | 观察性过程变量，不识别访谈者因果行为。 |
| `R` 扩展块 | 位置、密度、三段计数、转移、连续运行、路径深度、prompt-ID count/presence | B/C | 在可恢复性模块中更完整地表示“问了什么和路径如何展开”；变量按冻结注释生成。 | interviewer signal explanation plan | Rinaldi 的潜在提示类别不等于这些手工结构变量。 |
| block-Shapley | P/Q/R/D 的 24 种进入顺序全部平均 | C | 对 4 个块的顺序依赖进行对称汇总，并满足效率核验。 | interviewer signal explanation plan | 是描述性恢复度分配，不是因果贡献或信号百分比。 |

### 2.6 校准、阈值和指标

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| 主排序指标 | ROC-AUC | B | 与固定阈值无关，适合比较来源加入前后的排序变化；所有成对模型共享参与者。 | 分析计划与 manifests | AUC 不证明概率可用或临床效用。 |
| 支持指标 | PR-AUC、Brier、log loss、Macro-F1、sensitivity、specificity | B | 分离不平衡队列下的阳性排序、概率质量和操作点表现。 | 分析计划 | 支持指标不与主 AUC 家族混为一个结论。 |
| 默认操作点 | `0.50` | B | 常规概率分类默认值，用作透明基线。 | `model_configuration.md` | 明确不是临床 cutoff 或预注册阈值。 |
| 折内阈值规则 | max Macro-F1；max Youden J；sensitivity subject to specificity ≥0.80 | B | 从不同决策目标检查 0.50 失配；只在外层训练 OOF 概率上选择。 | `analysis_v2/00_plan/analysis_plan_v2.md` | 不能使用外层测试标签选阈值，也不声称某规则临床最优。 |
| 主校准 | Platt scaling | B | 参数化、在小样本中相对稳定；校准器仅拟合外层训练的交叉拟合概率。 | `model_configuration.md` | 校准后性能仍是内部交叉验证结果。 |
| 补充校准 | Isotonic | B | 允许非参数单调关系，用作模型形式敏感性；小样本下明确可能不稳定。 | post-audit sensitivity plan | 不因一个 aggregate Brier 较低就称其“最佳”。 |
| Brier 无技能基线 | 每个外层训练折阳性率 | B | 避免用全队列患病率给对应测试折造成信息渗透。 | post-audit sensitivity plan | `43/142` 只能作描述性全队列参照。 |

### 2.7 不确定性、检验和多重比较

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 冻结与溯源 | 解释边界 |
|---|---|---|---|---|---|
| participant bootstrap | 5,000 replicates | D | 在可接受计算量内稳定估计 percentile 95% CI；成对比较共同重采样同一参与者。 | `model_configuration.md`；各正式 manifest | bootstrap 次数不是发表文献要求；CI 仍受 n=142 限制。 |
| participant swap permutation | 10,000 swaps | D | 为参与者级成对差异提供约 `1/(10000+1)` 的 Monte Carlo p 值分辨率；计算上可行并在运行前冻结。 | `reanalysis_v2/inference.py`；manifests | p 值精度不等于效应估计精度，也不修复设计偏倚。 |
| 正式配对 exact sign-flip | 10 个 repeat 均值的全部符号翻转 | D | repeat 数仅为 10 时可穷举 `2^10` 个符号组合，无需 Monte Carlo；避免把 50 draws 当独立样本。 | `reanalysis_v2/identification_formal.py`；formal manifest | 最小双侧 p 值为 `2/2^10=0.001953125`，推断分辨率受 10 个 repeat 限制。 |
| FDR | Benjamini–Hochberg，预先定义家族 | D | 控制同一研究问题族中的多重比较，同时避免把所有主次分析混成一个家族。 | 分析计划、formal manifests | q 值只对其所属家族有效；不能跨模块比较 q 的大小。 |
| 主条件增量 FDR | late-fusion L4 的 MPNet/BGE 两行；另报告全部 late L1–L4 八行家族 | D | 将最完整控制后的核心问题与层级敏感性分开。 | embedding incremental plan | early concat 探索性行不被偷偷加入主家族。 |
| 正式配对/Fake-D FDR | 2 embeddings × 3 contrasts = 6 rows | D | 六行共享同一识别问题和正式推断层级。 | formal pairing manifest | 不把补充 PHQ 分区或 late fusion 结果加入该主家族。 |

### 2.8 Random、Matched、Fake-D 与文本窗口 draws

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| 正式表示和融合 | MPNet、BGE；early concat only | C/D | 两种冻结 dense 表示可直接在特征层破坏 P–I 配对；限制为一种融合避免识别实验组合爆炸。 | formal pairing manifest | late fusion 和 PHQ 分区只能作补充。 |
| repeats | 10 | D | 使用既有全部重复划分量化 split 稳定性，并形成 repeat-level contrasts。 | split 文件；formal manifest | repeat 不是独立受试者。 |
| Random draws | 每 repeat 50 | D | 对不同无固定点、标签盲随机配对取平均，降低单次随机错配偶然性；在正式运行前冻结且计算可行。 | formal manifest 与 ledger | 50 draws 不把推断样本扩成 500；推断单位是 10 个 repeat 均值。 |
| Matched draws | 每 repeat 50 | D | 对匹配中的微小随机 tie-breaking 和不同 donor mapping 做分布化评估。 | formal manifest 与 pairing ledger | Matched 只控制列出的可观察结构。 |
| Matched 变量 | interviewer word count、interviewer question count、clinical question count、unique protocol prompt count、clinical prompt coverage count、non-explicit interviewer turn count | C/D | 针对词量、提问量和协议覆盖这些最直接替代解释做一对一最小距离匹配；对角线禁止。 | formal pairing manifest；`reanalysis_v2/identification_sensitivity.py` | 不包括所有对话语义或未测变量，不能称完全匹配。 |
| Fake-D draws | 每 repeat 50 | D | 形成多个标签盲 D-P 完整行置换，评估真实患者—症状对应相对一般特征竞争的特异性。 | formal manifest 与 fake-D ledger | 不是生成“假症状文本”，也不能证明机制。 |
| draw 共享 | draw IDs/seeds 在 MPNet 与 BGE 间共享 | D | 使表示间比较面对相同扰动，实现成对、可审计比较。 | closeout amendment 与 formal manifest | 共享 draw 不使两个 embedding 结果统计独立。 |
| 文本预算随机窗口 | 50 draws | D | 随机化窗口起点，避免只选择某个有利位置；点估计对 50 个 draw-specific AUC 取均值。 | post-audit sensitivity plan | 只是所测 whitespace-token 预算下的敏感性。 |
| half-min 预算 | 双方各取 `floor(0.5 × min(P words, I words))` | C/D | 对双方施加完全相同且严格小于较短来源的预算，避免原不对称截断保留一方全文。 | post-audit sensitivity plan | 不能代表所有可能的长度归一化方案。 |
| 位置窗口 | early、middle、late | C/D | 检查提示区域和访谈阶段是否影响来源差异，不用结果确定窗口。 | post-audit sensitivity plan | 是确定性位置敏感性，不是时间因果效应。 |

### 2.9 D-P 人工效度验证参数

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| 主验证样本 | 30 人 | D | 按标签×参与者词数三分位形成 6 个层，每层 5 人；在可行人工工作量下保留主要队列结构。 | blind quality audit packet 3；claim-evidence matrix | 不是效应量驱动的全队列 power sample，也不是全部 142 人金标准。 |
| 主判定单元 | 参与者×10 域，共 300 个主样本域单元 | D | presence 和 count 的目标天然位于参与者—症状域层级；阳性与零值域都需查看原文。 | stage-1 protocol/codebook | 不能只审核模型抽出的阳性片段。 |
| 稀有域富集 | 13 人，单独报告 | D | 补足主样本中稀有阳性或关键零值域，描述错误类型；不污染总体代表性估计。 | packet 3 与 formal closeout docs | 不与 30 人直接混算总体准确率，除非使用明确抽样权重。 |
| 哨兵个案 | 1 名预先指定，单独报告 | D | 检查全域零值究竟是无证据、明确否认、覆盖不足还是整体漏抽。 | packet 3 protocol | 不进入主要效度统计；公开材料不保留真实 participant ID。 |
| 评价者 | 2 名独立评价者 | D | 使人工参照具有可量化一致性，并在共识前保持独立判断。 | Stage-1 A/B forms 与 frozen consensus | 两人一致不等于临床金标准。 |
| 盲法阶段 | Stage 1 不看 C5/D-P/PHQ/模型结果；共识冻结后才进入 Stage 2 | D | 防止自动结果锚定人工参照，并区分人工真值构建与自动证据审核。 | stage-1 protocol；stage-2 gate | Stage-2 结构检查不等于最终证据遗漏率已裁决。 |
| presence 主映射 | 当前阳性映射为 positive；其他非冲突状态为 non-current-positive；conflicting 主分析排除并作敏感性 | D | 主要目标是当前症状 presence，冲突证据不应强制塞入单一主类别。 | frozen codebook/analysis contract | 非当前阳性不等于“从未有症状”。 |
| count | `0/1/2/3+`，次要结果 | D | 限制人工和自动计数对极端多条证据的敏感性；presence 是主要效度目标。 | Stage-1 forms | count 相关高不能替代域级临床效度。 |
| 不确定性 | 参与者聚类 bootstrap | D | 10 个域嵌套在同一参与者内，必须共同重采样参与者而非把 300 单元视为独立。 | scoring contract | 域级单元不是 300 个独立受试者。 |

### 2.10 随机种子、哈希和重复计算

| 参数/选择 | 正式值 | 类型 | 为什么这样选 | 溯源 | 解释边界 |
|---|---|---|---|---|---|
| 主审计 base seed | `20260705` | D | 固定抽样、模型和推断随机过程，使结果可重算；数值本身没有科学含义。 | submission audit run manifest | 不能把某个 seed 的结果当一般规律，故另有 repeats/draws。 |
| embedding/pairing base seed | `20260624` | D | 在相关模块中统一生成划分内模型和扰动种子，并在 ledger/manifest 记录。 | embedding 与 formal pairing manifests | 不按结果更换 seed。 |
| 人工主样本 seed | `20260714` | D | 冻结分层抽样的可重复选择；富集和哨兵另行标记。 | packet 3 protocol | 随机种子不使富集样本具有总体代表性。 |
| Hub revision pinning | 完整 commit revision | C/D | 防止同名模型后续更新导致编码变化。 | embedding manifests | 复现需要相同模型文件和受限输入，不只是模型名相同。 |
| 输入/输出 SHA-256 | 每个正式输入、cache、输出和 manifest | D | 证明正式运行读取和产生的是冻结版本，并支持重复计算比较。 | 各 run manifests；`recomputation_verification.json` | 哈希一致证明文件一致，不证明研究设计或人工效度正确。 |

## 3. 参数选择的统一答辩逻辑

### 3.1 哪些参数可以说“文献直接支持”？

可以这样说的主要是研究对象和任务层级：PHQ-8 二分类操作化、参与者/访谈者分源、提示与协议语境、patient/therapist 或 question/answer 结构、症状级表示，以及专家/人工验证症状表示的必要性。

不能把 `50 draws`、`5,000 bootstrap`、`C=1` 或 `200-word chunks`说成某篇 DAIC-WOZ 论文规定的标准值。

### 3.2 为什么 TF-IDF 的 C 固定，而 dense 的 C 折内选择？

TF-IDF `C=1.0` 属于已经冻结的主审计配置，继续调参会改变原主结果。Dense 表示维数和几何性质不同，因此在运行前固定一个紧凑对数网格，并只在外层训练折内部选择。两者的共同红线是外层测试折不参与优化。

### 3.3 为什么是 10 repeats，而不是更多？

10 repeats 是现有冻结 split 合同的一部分，主要作用是检查划分稳定性。正式配对分析选择 repeat-level contrast 作为推断单位，避免把同一队列上的 50 draws 或 50 folds伪装成独立样本。代价是 exact sign-flip 的 p 值分辨率较粗，应连同效应方向和区间报告。

### 3.4 为什么每类配对做 50 draws？

单次错配容易受某个 donor mapping 偶然影响。50 draws 用于平均扰动随机性，并在计算预算内让 draw 均值趋于稳定；它们不是增加样本量。参数在正式运行前冻结，推断仍以 10 个 repeat 的均值为单位。

### 3.5 为什么 bootstrap 5,000、permutation 10,000？

二者解决不同问题：5,000 bootstrap 用于区间估计；10,000 成对 swap 用于零假设下的 Monte Carlo p 值。该数量在当前 142 人和模型输出上计算可行，且给出足够的数值分辨率。它们提高数值稳定性，但不能弥补队列小、单数据集或观察性设计的限制。

### 3.6 为什么只用 MPNet 和 BGE 做正式配对？

配对破坏需要在特征层同时保留 P 与 I，因此选用已经冻结、缓存、哈希验证的两种 dense 表示，并固定 early concat。这样检验表示稳健性又控制计算规模；不是因为这两种模型在结果出来后 AUC 最高。

### 3.7 为什么不继续加模型、BERTopic 或网络分析？

当前研究问题已经由单来源、条件增量、可恢复性、配对破坏、Fake-D 和人工验证覆盖。继续增加模型或网络分析会扩大研究问题，却不会补上现阶段最关键的“为什么这样设计”的证据链。图和主题网络可作为后续独立研究，不作为本稿缺失的参数验证。

## 4. 参数冻结时点与证据等级

| 内容 | 时间性质 | 正式表述 |
|---|---|---|
| 原 10×5 split、主输入与基础模型配置 | 既有分析后冻结并复用 | frozen/analysis-locked；不得称原始预注册 |
| 投稿审计 repeat 1、校准与文本预算计划 | 投稿审计前或对应敏感性运行前锁定 | pre-result frozen for that audit/sensitivity；不是整项研究预注册 |
| embedding incremental L1–L4 计划 | 补充分析运行前锁定 | post hoc, analysis-locked supplementary analysis |
| 正式 10-repeat pairing/Fake-D closeout amendment | repeat-1 探索后、正式 10-repeat 运行前锁定 | formal follow-up to an exploratory signal；不得称原始假设预注册 |
| D-P 人工验证 packet 3 | 正式人工编码前冻结 | pre-review protocol；人工结果不反向修改 codebook |

## 5. 允许和禁止的总括表述

### 允许

> 参数分为任务定义、折内选择、表示稳健性和推断/计算预算四类。所有可调预处理和模型选择均限制在外层训练数据内；表示、扰动次数、推断次数、FDR 家族和随机种子在相应正式运行前冻结并写入 manifest。多种表示和对照用于检验结论的操作化敏感性，而非筛选最高性能模型。

### 禁止

- “所有参数都有文献证明是最优值”；
- “10 repeats 或 50 draws 等于增加了独立样本量”；
- “5,000 bootstrap 使小样本问题消失”；
- “Matched 已控制全部混杂”；
- “Fake-D 证明症状机制”；
- “AUC、校准或阈值结果证明临床可用”；
- “analysis-locked 等同于 preregistered”。

## 6. 主要内部溯源文件

- `analysis_v2/00_plan/analysis_plan_v2.md`
- `analysis_v2/00_plan/changelog.md`
- `analysis_v2/00_splits/repeated_5fold_splits_10x5.csv`
- `analysis_v2/03_embedding_robustness/model_1_all_mpnet_base_v2/run_manifest.json`
- `analysis_v2/03_embedding_robustness/model_2_bge_large_en_v1_5/run_manifest.json`
- `analysis_v2/03_embedding_robustness/embedding_incremental_analysis_plan.md`
- `analysis_v2/03_embedding_robustness/embedding_incremental/final/run_manifest.json`
- `analysis_v2/10_source_importance/08_identification_sensitivity/run_formal_10x/run_manifest.json`
- `analysis_v2/12_submission_audit/model_configuration.md`
- `analysis_v2/12_submission_audit/post_submission_audit_sensitivity_plan.md`
- `analysis_v2/12_submission_audit/run_manifest.json`
- `analysis_v2/15_interviewer_signal_explanation/interviewer_signal_explanation_plan.md`
- `reanalysis_v2/modeling.py`
- `reanalysis_v2/embedding_incremental.py`
- `reanalysis_v2/identification_sensitivity.py`
- `reanalysis_v2/identification_formal.py`

## 7. 已核验的外部方法依据

1. Rinaldi, A., Fox Tree, J., & Chaturvedi, S. (2020). Predicting Depression in Screening Interviews from Latent Categorization of Interview Prompts. https://doi.org/10.18653/v1/2020.acl-main.2
2. Burdisso, S., et al. (2024). DAIC-WOZ: On the Validity of Using the Therapist’s Prompts in Automatic Depression Detection from Clinical Interviews. https://doi.org/10.18653/v1/2024.clinicalnlp-1.8
3. Agarwal, N., Dias, G., & Dollfus, S. (2024). Analysing Relevance of Discourse Structure for Improved Mental Health Estimation. https://doi.org/10.18653/v1/2024.clpsych-1.9
4. Milintsevich, K., Sirts, K., & Dias, G. (2023). Towards Automatic Text-based Estimation of Depression through Symptom Prediction. https://doi.org/10.1186/s40708-023-00185-9
5. Agarwal, N., et al. (2024). Analyzing Symptom-based Depression Level Estimation through the Prism of Psychiatric Expertise. https://aclanthology.org/2024.lrec-main.87/
6. Agarwal, N., Dias, G., & Dollfus, S. (2024). Multi-view Graph-based Interview Representation to Improve Depression Level Estimation. https://doi.org/10.1186/s40708-024-00227-w

## 8. 最终结论

本研究参数体系的合理性不依赖“别人也用了同一个数字”，而依赖四条可审计原则：

1. 任务和来源定义有直接的 DAIC-WOZ 实证基础；
2. 可调模型参数只在外层训练数据内选择；
3. 表示和对照在正式运行前冻结并全部报告；
4. draws、bootstrap、permutation 和 FDR 明确对应不同的不确定性与检验目标，且不被误当成新增独立样本。

因此，参数依据可以正式表述为“文献支持研究层面，训练折内选择模型参数，运行前冻结敏感性与推断参数，manifest 证明实际执行”，而不能表述为“所有数值都由前人论文给出”。
