# DAIC-WOZ 分析文件索引

本目录是为了方便人工查看而整理的中文命名副本。原始文件没有改名，脚本仍然使用原路径。

## 缩写规则

- A = 分析源数据
- B = 模型结果与偏差指标
- C = 论文表格
- D = 论文图
- E = 附录与稳健性材料
- S = 复现脚本

## A 分析源数据

| 文件 | 用途 |
|---|---|
| `A1_样本索引_标签人口学与转录统计.csv` | 表 1 的主要来源；包含 participant_id、split、label、PHQ-8、性别、年龄、PTSD、轮次和词数统计。 |
| `A2_输入条件长表_C1至C4推荐分析版.csv` | 模型训练和表 2 的主要来源；包含 C1 full dialogue、C2 participant-only、C3 interviewer-only、C4 interviewer-cleaned。 |
| `A3_质量标记_访谈者缺失样本.csv` | 记录 451、458、480 没有 interviewer turns；C3/C4 分析需要排除或报告。 |
| `A4_转录统计_轮次词数诊断问题.csv` | 转录统计备查；用于核对访谈轮次、诊断性问题数和文本长度。 |
| `A5_标签表_PHQ8二分类.csv` | 标签备查；PHQ-8 >= 10 为抑郁阳性。 |
| `A6_数据划分_train_dev.csv` | train/dev 划分备查。 |

## B 模型结果与偏差指标

| 文件 | 用途 |
|---|---|
| `B1_固定train_dev模型性能汇总.csv` | 固定 train/dev 划分下的模型性能。 |
| `B2_五折交叉验证模型性能汇总.csv` | 5-fold CV 下的模型性能；表 3 和图 D1 的主要来源。 |
| `B3_偏差指标_IBPG_DPD_F1_MacroF1_MCC.csv` | IBPG 和 DPD 的机器可读结果。 |
| `B3_偏差指标_IBPG_DPD_F1_MacroF1_MCC.md` | IBPG 和 DPD 的 Markdown 阅读版。 |
| `B4_图源数据_MacroF1_MCC五折CV.csv` | D1 Macro-F1/MCC 条形图的 source data。 |
| `B5_阈值敏感性_五折CV_MCC调阈值.csv` | TF-IDF + LR/SVM 的 5-fold CV MCC 调阈值汇总。 |
| `B5_阈值敏感性_五折CV_MCC调阈值.md` | B5 的 Markdown 阅读版。 |
| `B6_McNemar配对检验_train_dev.csv` | 固定 train/dev 划分下，不同输入条件之间的 McNemar 精确检验。 |
| `B6_McNemar配对检验_train_dev.md` | B6 的 Markdown 阅读版。 |
| `B7_图源数据_阈值敏感性五折CV.csv` | D2 阈值敏感性图的 source data。 |
| `B8_阈值敏感性_逐折明细_MCC调阈值.csv` | 调阈值实验的逐折明细，便于复查每折阈值和性能。 |
| `B9_阈值稳健性_nested_oracle五折CV.csv` | TF-IDF + LR/SVM 的 nested threshold 与 oracle threshold 补充分析汇总。 |
| `B9_阈值稳健性_nested_oracle五折CV.md` | B9 的 Markdown 阅读版。 |
| `B10_阈值稳健性_逐折明细_nested_oracle.csv` | nested/oracle threshold 补充分析的逐折明细和阈值信息。 |

## C 论文表格

| 文件 | 用途 |
|---|---|
| `C0_论文表格总表_C1至C4.xlsx` | 早期表 1-4 的 Excel 总表副本。 |
| `C0_论文表格总表_C1至C4.md` | 早期表 1-4 的 Markdown 合并版副本。 |
| `C0_论文表格总表_C1至C4_含阈值敏感性与McNemar.xlsx` | 当前推荐总表；包含表 1-4、阈值敏感性表和 McNemar 检验表。 |
| `C0_论文表格总表_C1至C4_含阈值敏感性与McNemar.md` | 当前推荐总表的 Markdown 合并版。 |
| `C1_表1_样本基本特征.csv/md` | 表 1：样本基本特征。 |
| `C2_表2_输入条件文本长度.csv/md` | 表 2：C1-C4 输入条件文本长度。 |
| `C3_表3_模型输入条件性能_五折CV.csv/md` | 表 3：模型 × 输入条件性能，5-fold CV。 |
| `C4_表4_偏差指标_IBPG_DPD.csv/md` | 表 4：IBPG / DPD 偏差指标。 |
| `C5_表5_阈值敏感性_五折CV.csv/md` | 表 5：默认阈值与 MCC 调阈值的性能对照。 |
| `C6_表6_McNemar配对检验_train_dev.csv/md` | 表 6：固定 train/dev 划分下的 McNemar 配对检验。 |
| `C7_补充表_阈值稳健性_nested_oracle五折CV.csv/md` | 补充表：nested threshold 与 oracle threshold 的稳健性结果。 |

## D 论文图

| 文件 | 用途 |
|---|---|
| `D1_MacroF1_MCC条形图_五折CV.png` | 论文/PPT 可直接查看的位图。 |
| `D1_MacroF1_MCC条形图_五折CV.svg` | 可编辑矢量图。 |
| `D1_MacroF1_MCC条形图_五折CV.pdf` | 投稿或排版用矢量 PDF。 |
| `D2_阈值敏感性_MacroF1_MCC五折CV.png` | 默认阈值 vs MCC 调阈值差值图；零差值用空心圆标记。 |
| `D2_阈值敏感性_MacroF1_MCC五折CV.svg` | D2 的可编辑矢量图。 |
| `D2_阈值敏感性_MacroF1_MCC五折CV.pdf` | D2 的排版用矢量 PDF。 |

## E 附录与稳健性材料

| 文件 | 用途 |
|---|---|
| `E1_C4清洗规则与审计附录.md` | C4 interviewer-cleaned 的清洗规则、规则类别、自动审计结果和人工抽查说明。 |
| `E2_C4清洗规则目录.csv` | C4 清洗规则类别的机器可读目录。 |
| `E3_C4清洗审计摘要.csv` | C4 清洗前后保留/删除轮次和词数的审计摘要。 |
| `E4_C4人工抽查模板.csv` | 人工抽查模板；`human_review_decision` 和 `reviewer_notes` 留空，供后续人工判定。 |

## S 复现脚本

| 文件 | 用途 |
|---|---|
| `S1_传统基线运行脚本_TFIDF_MPNet_LR_SVM.py` | 运行 TF-IDF + LR、TF-IDF + SVM、MPNet + LR 的基线脚本。 |
| `S2_论文表格和图生成脚本.py` | 从分析结果生成表 1-6、D1 和 D2 的脚本。 |
| `S3_阈值敏感性和McNemar检验脚本.py` | 运行 TF-IDF 阈值敏感性分析和 McNemar 配对检验的脚本。 |
| `S4_附录与阈值稳健性分析脚本.py` | 生成 C4 清洗附录、人工抽查模板，以及 nested/oracle threshold 补充分析。 |

## 当前研究主分析文件

主分析优先看这几份：

1. `A2_输入条件长表_C1至C4推荐分析版.csv`
2. `B2_五折交叉验证模型性能汇总.csv`
3. `B3_偏差指标_IBPG_DPD_F1_MacroF1_MCC.csv`
4. `B5_阈值敏感性_五折CV_MCC调阈值.csv`
5. `B6_McNemar配对检验_train_dev.csv`
6. `B9_阈值稳健性_nested_oracle五折CV.csv`
7. `C0_论文表格总表_C1至C4_含阈值敏感性与McNemar.xlsx`
8. `C7_补充表_阈值稳健性_nested_oracle五折CV.csv`
9. `D1_MacroF1_MCC条形图_五折CV.png`
10. `D2_阈值敏感性_MacroF1_MCC五折CV.png`
11. `E1_C4清洗规则与审计附录.md`
12. `E4_C4人工抽查模板.csv`
13. `论文前半部分工作稿_研究逻辑与方法_20260615.md`

## 注意

输入条件 C5 `symptom_evidence_only` 还没有进入当前 C1-C4 主分析文件。这里的 `C5_表5_阈值敏感性_五折CV` 是论文表格编号，不是输入条件 C5。后续完成 LLM 症状证据抽取后，应新增一组 `A7/C5症状证据输入` 相关文件。

当前标题已暂时收紧为“临床访谈文本输入条件对抑郁识别模型的影响”，不再把“大语言模型”作为主标题核心；除非后续实际运行 API LLM 或症状证据抽取模型，否则主文应继续按传统文本分类与输入条件稳健性来写。
