# 重新分析总方案 v2

## 一、核心原则与文章主线

### 核心原则

本研究后续重新分析不以“把不显著做显著”为目标，而以“检验输入来源效应是否依赖表征方式、阈值策略和证据浓缩方式”为目标。无论最终结果是否显著，论文均定位为方法学审计研究，重点揭示输入来源、模型表征、概率校准和阈值决策对 DAIC-WOZ PHQ-8 预测结论的影响。

### 文章新主线

本研究审计 DAIC-WOZ 脚本化访谈中的输入来源效应，重点区分并检验三件事：

1. 参与者言语、访谈者言语和症状证据片段在风险排序能力（ranking）上是否不同；
2. 不同输入来源是否导致默认阈值下的决策表现差异；
3. 这些差异是否依赖 TF-IDF 这类浅层词频表征，还是在 dense embedding 等深层语义表征下仍然存在。

---

## 二、总体分析模块与优先级

| 模块 | 内容 | 优先级 | 目的 |
|---|---|---:|---|
| 模块 1 | 复现并冻结当前主分析 | 最高 | 固定数据、标签、split、输入条件和 TF-IDF 主分析 |
| 模块 2 | 阈值与概率校准分析 | 最高 | 判断参与者言语默认阈值低灵敏度的来源 |
| 模块 3 | Dense embedding 稳健性分析 | 高 | 检验输入来源效应是否依赖浅层词频表征 |
| 模块 4 | C5 证据浓缩负对照 | 中高 | 排除“仅仅是词数压缩/去噪”的替代解释 |
| 模块 5 | C4 访谈者模板机制控制 | 中高 | 分离模板、词量、位置和协议结构的贡献 |

---

## 三、模块 1：复现并冻结当前主分析

### 1.1 固定数据与标签

- 数据范围：官方 train + dev 共 142 人。
- 官方 test：47 人，暂时仅作为 exploratory sensitivity，不作为独立验证。
- 标签定义：PHQ-8 ≥ 10 为阳性。
- 当前标签分布：阳性 43 人，阴性 99 人。

### 1.2 固定交叉验证 split

- CV 策略：10 次重复 × 5 折分层交叉验证。
- 强制要求：保存每一次 repeat、fold 的 participant ID 列表至：

```text
analysis_v2/00_splits/repeated_5fold_splits_10x5.csv
```

后续所有模型和消融实验，包括 TF-IDF、dense embedding、C4 控制、C5 负对照，必须强制使用同一套 split，以确保配对检验可比。

### 1.3 固定五个主输入条件

| 条件编号 | 条件名称 | 说明 |
|---|---|---|
| C1 | full_transcript | 完整访谈转录，包含参与者与访谈者 |
| C2 | participant_speech | 仅参与者言语 |
| C3 | interviewer_speech | 全部访谈者言语 |
| C4 | non_explicit_interviewer_speech | 删除显性临床提问后的访谈者言语 |
| C5 | participant_symptom_evidence | 人工复核后的参与者症状证据片段 |

---

## 四、模块 2：阈值与概率校准分析

本模块优先执行，用于检验 C2 参与者言语默认阈值下灵敏度极低的可能来源。分析目标不是预设证明“分数压缩”，而是区分以下可能性：

1. 默认 0.5 阈值失配；
2. 概率校准不足；
3. 当前表征和分类框架下参与者言语难以转化为稳定筛查决策。

### 2.1 三种决策方式比较

#### 2.1.1 默认阈值

- 固定 threshold = 0.5。

#### 2.1.2 嵌套阈值

- 外层：原 10×5 repeated stratified CV。
- 内层：每个外层训练折内部做 3-fold CV。
- 在内层 OOF 概率上选择阈值，然后应用于外层测试折。
- 预设三种阈值规则：

| 阈值规则 | 报告位置 | 定义 |
|---|---|---|
| max_macro_f1 | 主文 | 选择使 Macro-F1 最大的阈值 |
| max_youden_j | 补充 | 选择使 sensitivity + specificity − 1 最大的阈值 |
| sensitivity_at_spec80 | 补充 | 在 specificity ≥ 0.80 条件下选择 sensitivity 最大的阈值 |

#### 2.1.3 校准后阈值

- 主校准方法：Platt scaling。
- 补充校准方法：Isotonic calibration。
- 校准器仅在外层训练折中拟合。
- 校准后的测试折概率再使用 threshold = 0.5 进行分类。
- Isotonic calibration 因小样本过拟合风险较高，仅作为补充稳健性分析。

### 2.2 强制报告的指标

所有 C1–C5 条件均须报告以下指标：

| 类型 | 指标 |
|---|---|
| 排序能力 | ROC-AUC, PR-AUC |
| 决策表现 | Macro-F1@0.5, Macro-F1@nested, Sensitivity@0.5, Sensitivity@nested, Specificity@0.5, Specificity@nested |
| 校准质量 | Brier score, 5-bin quantile ECE, calibration slope/intercept |
| 分数分布 | 阳性/阴性预测概率均值、标准差、中位数、IQR |

说明：预测概率均值与标准差用于量化描述分数分布和分数压缩现象，不单独作为因果证明。

### 2.3 嵌套阈值稳定性

补充材料中报告 50 个外层 fold 选出的阈值分布，包括：mean、SD、median、IQR、min/max，以及 density plot 或 boxplot。

如果 C2 的阈值分布高度波动，应解释为该输入来源在当前模型下概率尺度不稳定，而不是直接解释为临床信号缺失。

### 2.4 结果解释决策树

| 结果 | 解释 |
|---|---|
| C2 在 nested threshold 后明显恢复，例如 sensitivity 接近或超过 0.50 | 参与者言语并非缺乏预测信号，主要问题是默认阈值失配 |
| C2 校准后恢复，表现为 SD 扩大、sensitivity 恢复 | 默认阈值失败部分源于概率校准不足，不能简单解释为参与者语言信号较弱 |
| C2 nested threshold 和校准后仍差 | 在该表征和模型框架下，参与者言语不仅阈值失配，且可转化为稳定筛查决策的信号有限 |

---

## 五、模块 3：Dense embedding 稳健性分析

本模块用于检验输入来源效应是否依赖浅层词频表征。该模块不是为了追求最高性能，而是用于检验 TF-IDF 主分析结论的表征稳健性。

### 3.1 固定 embedding 模型

DAIC-WOZ 为英文数据，禁止使用中文 embedding 模型。预注册固定两个英文或多语种 dense embedding 模型。建议最终在代码执行前从以下候选中明确锁定两个，不允许跑多个模型后选择性报告。

| 类别 | 候选模型 | 说明 |
|---|---|---|
| 轻量稳健 | all-mpnet-base-v2 或 bge-large-en-v1.5 | 通用英文句向量模型 |
| 强模型 | gte-Qwen2-7B-instruct 或 Qwen embedding 系列 | 大模型系 embedding |

冻结执行版本建议：

```text
Dense embedding model 1: all-mpnet-base-v2
Dense embedding model 2: gte-Qwen2-7B-instruct
```

如果因显存或运行环境限制无法运行 gte-Qwen2-7B-instruct，应在执行日志中记录，并预先替换为 bge-large-en-v1.5 或 bge-m3。

### 3.2 文本处理与 participant-level embedding

长文本不能直接截断为 512 token。采用以下流程：

1. 按 utterance 或 512-token chunk 切分文本；
2. 对每个 utterance/chunk 获取 embedding；
3. 对同一 participant、同一输入条件下的所有 utterance/chunk embedding 做 mean pooling；
4. 得到 participant-level vector。

默认聚合方式为 mean pooling。若使用 length-weighted mean pooling，必须作为补充分析，并在所有输入条件中一致使用。

### 3.3 分类器与正则化

分类器严格保持与 TF-IDF 分析一致的模型家族：

```text
LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    max_iter=2000
)
```

为防止高维 embedding 在 142 人样本上过拟合，不做全局 PCA。优先使用 L2 正则，并在每个外层训练折内部通过 inner CV 调节：

```text
C ∈ [0.01, 0.1, 1.0, 10.0]
```

所有 C 参数选择必须仅使用外层训练折，不能使用外层测试折。

### 3.4 评估规则

Dense embedding 模型必须与 TF-IDF 使用完全相同的：10×5 OOF split、participant-level OOF 汇总、配对置换检验、BH-FDR 校正、默认阈值、嵌套阈值、Platt calibration 和补充 isotonic calibration。

### 3.5 结果解释决策树

| 结果 | 解释 |
|---|---|
| Dense embedding 下 C3 > C2 的 AUC 变得显著 | 访谈者侧优势在浅层词频下主要表现为阈值决策差异，在深层语义下进一步表现为排序能力差异，提示协议高阶语义结构被深层模型放大 |
| Dense embedding 下 C2 和 C3 都变强，但 C3 > C2 仍不显著 | TF-IDF 可能低估参与者语义信号，访谈者优势主要稳定存在于阈值决策层面，而非跨表征稳定的排序优势 |
| Dense embedding 下 C2 超过 C3 | TF-IDF 下的参与者弱表现可能是表征限制导致的，深层模型能恢复参与者临床语义信号 |
| 所有结果都不显著 | 未发现跨表征稳定的访谈者排序优势，但默认阈值决策指标仍可能对输入来源高度敏感 |

---

## 六、模块 4：C5 证据浓缩负对照

本模块用于保护或降级“证据浓缩有效”的结论，排除“仅仅是压缩词数/去噪”的替代解释。

### 4.1 新增 C5 对照条件

| 条件 | 构造方式 | 目的 |
|---|---|---|
| C5-reviewed-evidence | 当前人工复核症状证据 | 核心条件 |
| C5-random-same-word | 从参与者言语中随机抽同词数文本 | 控制词数与压缩 |
| C5-nonsymptom-same-word | 抽取非症状片段并匹配词数 | 控制“非症状但同样压缩”的文本 |
| C5-domain-presence | 只用 10 个症状域是否存在，二值向量 | 判断是否主要来自症状域覆盖结构 |
| C5-domain-count | 只用每个症状域 quote 数量 | 判断是否主要来自症状域数量结构 |
| C5-keyword-masked | 遮蔽 sleep/tired/depressed 等高显性词 | 判断是否依赖少数显性关键词 |

### 4.2 随机负对照要求

C5-random-same-word 必须跑 10 个不同随机种子：

```text
C5_random_same_word_seed_01
...
C5_random_same_word_seed_10
```

每个 seed 均跑完整 10×5 CV。最终报告 mean ROC-AUC、SD、95% percentile interval、C5-reviewed-evidence vs random distribution 的比较。所有随机种子必须保存。

### 4.3 关键解释限制

如果 C5-domain-presence 表现接近 C5-reviewed-evidence，绝对不能解释为“问了没大于说了啥”或“临床路径执行效应”。

唯一允许的解释写法：

> 症状域覆盖结构，即“抽取到的症状域证据状态”，可能已携带大量判别信息，具体 quote 文本的额外语义贡献有限。

只有在额外编码 Ellie 是否询问某症状域之后，才允许讨论“访谈者提问路径”或“问了没”的机制。

---

## 七、模块 5：C4 访谈者模板机制控制

本模块用于澄清 C4 消融中“模板移除与随机移除无差异”的机制，分离模板、词量、位置和协议结构的贡献。

### 5.1 新增 C4 控制条件

| 条件 | 构造方式 | 目的 |
|---|---|---|
| C4-nontemplate-lengthmatched | 从非模板文本中抽取与模板同等词数 | 控制词量 |
| C4-nontemplate-positionmatched | 从模板邻近位置抽取同词数非模板话轮 | 控制访谈阶段/位置 |
| C4-template-shuffled | 模板文本在被试之间随机置换 | 判断是否具有个体特异性 |
| C4-template-presence | 只用模板 ID 是否出现/出现次数 | 判断是否是协议覆盖或次数信号 |

### 5.2 关键比较

| 比较 | 若显著 | 若不显著 |
|---|---|---|
| template-only vs lengthmatched | 支持模板文本超出词量的额外信息 | 说明主要可能来自词量或文本量 |
| template-only vs positionmatched | 支持模板语义或模板结构 | 说明主要可能来自访谈阶段/位置 |
| template-only vs shuffled | 支持个体特异模板信息 | 若不变，说明可能是协议结构或覆盖模式 |
| template-text vs template-presence | 支持自然语言文本内容 | 若接近，说明可能主要来自覆盖/次数 |

### 5.3 统一解释写法

不管各子条件结果如何，C4 部分最终解释必须统一为：

> 访谈者侧信号并非仅来自粗粒度长度计数，但当前结果也不能将其特异性归因于模板自然语言语义。模板、位置、话题覆盖和词量共同构成脚本化访谈中的协议信号。

---

## 八、统一统计检验规则

### 8.1 基准

所有比较均基于 participant-level OOF 概率，即每名参与者 10 次重复预测概率的均值。严禁折均值与 participant-level OOF 混用。

### 8.2 配对置换检验流程

1. 每名参与者有条件 A 和条件 B 的 OOF 概率；
2. 每次置换在参与者层面随机交换 A/B；
3. 重新计算目标指标差值；
4. Monte Carlo 置换 10,000 次；
5. 计算双侧 p 值；
6. 使用 Benjamini–Hochberg 方法进行 FDR 校正。

### 8.3 严格 FDR 分族

禁止跨模块混算 FDR。固定分为 4 个 family：

| Family | 包含比较 | 报告位置 |
|---|---|---|
| Primary input-source comparisons | C1–C5 之间的主条件比较 | 主文 |
| Representation robustness | TF-IDF vs embedding 的核心比较 | 主文 |
| C5 evidence controls | C5 与随机、非症状、域存在等比较 | 补充材料，主文引用方向性结论 |
| C4 interviewer controls | 模板、长度、位置、打乱等比较 | 补充材料，主文引用方向性结论 |

主文只重点解释 Family 1 和 Family 2。Family 3 和 Family 4 放补充材料，主文讨论中仅引用方向性结论。

---

## 九、最终执行顺序与时间线

### Step 0：冻结本分析计划

- 保存本文件。
- 后续不轻易更改规则。
- 若必须变更，例如模型无法运行、显存不足、文件缺失，必须在 changelog 中记录原因、日期和替代方案。

### Step 1：模块 1 + 模块 2

1. 冻结 splits；
2. 跑 TF-IDF C1–C5；
3. 计算 Brier、ECE、calibration slope/intercept、概率均值/SD；
4. 跑 nested threshold，主规则为 max_macro_f1；
5. 跑 Platt calibration；
6. 跑 isotonic calibration 作为补充。

### Step 2：模块 3

1. 跑预注册的 2 个英文或多语种 embedding 模型；
2. 仅跑 C1–C5；
3. 重复 Step 1 的所有统计、校准和阈值流程。

### Step 3：模块 4

1. 生成并跑 10 个随机种子的 C5-random-same-word；
2. 跑 C5-nonsymptom-same-word；
3. 跑 C5-domain-presence；
4. 跑 C5-domain-count；
5. 跑 C5-keyword-masked。

### Step 4：模块 5

1. 跑 C4-nontemplate-lengthmatched；
2. 跑 C4-nontemplate-positionmatched；
3. 跑 C4-template-shuffled；
4. 跑 C4-template-presence。

### Step 5：撰写论文

根据跑出的具体结果，对照各模块决策树组装可替换段落，最后定稿摘要与结论。

---

## 十、推荐输出目录结构

```text
analysis_v2/
  00_plan/
  00_splits/
  01_inputs/
  02_tfidf_main/
  03_embedding_robustness/
  04_c5_controls/
  05_c4_controls/
  06_tables_figures/
```

---

## 十一、最终解释红线

1. 不得把不显著的 ROC-AUC 差异写成显著优势。
2. 不得把默认阈值下的 sensitivity 差异直接解释为排序能力差异。
3. 不得把 C5-domain-presence 解释为“Ellie 是否询问该域”。
4. 不得把 C4 模板效应解释为已证明的模板自然语言语义效应。
5. 不得把 official test subset 写成独立外部验证。
6. 不得选择性报告 embedding 模型结果。
7. 不得将 LLM 证据抽取写成无需人工复核的最终诊断依据。
