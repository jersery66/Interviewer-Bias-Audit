# 表X DAIC-WOZ PHQ-8阳性预测中的信号来源分解

本表将DAIC-WOZ PHQ-8阳性预测中的信号分解为六个可解释的层级。这是一个描述性整合表，不是统计检验。

## 信号来源分解表

| 信号层级 | 操作化定义 | 对应分析 | 代表性结果 | 可以支持的结论 | 不能支持的结论 |
|---------|-----------|---------|-----------|--------------|--------------|
| 患者语言信号 | 仅使用患者言语（C2 participant_speech） | TF-IDF主分析 | AUC=0.705；Sensitivity@0.5=0.047；nested Sensitivity=0.512 | 患者言语中存在风险排序信号 | 不能说默认0.5阈值下可直接筛查 |
| 症状证据片段信号 | 人工复核后的症状quote（C5 reviewed evidence） | C5控制分析 | AUC=0.754；Macro-F1@0.5=0.682 | 症状证据浓缩后具有可用分类表现 | 不能说具体quote语义显著优于对照 |
| 症状域覆盖信号 | 症状域presence/count向量（domain_presence, domain_count） | C5控制分析（domain特征） | domain_presence AUC=0.790；domain_count AUC=0.794 | 症状域覆盖状态携带较强判别信息 | 不能说模型真正理解病理机制 |
| 访谈协议结构 | 访谈者文本、模板文本、模板presence（C3、C4、template controls） | C4控制分析 | template_only AUC=0.754；template_presence AUC=0.759 | 访谈流程与协议结构有可利用信号 | 不能说访谈者自然语言语义本身导致预测 |
| 表征方式 | TF-IDF、MPNet、BGE（三种表征方法） | 表征稳健性分析 | MPNet下C2>C3；BGE下C3>C2 | 输入来源效应依赖表征方式 | 不能说某个输入来源跨模型稳定最强 |
| 阈值策略 | 默认阈值（0.5）vs nested阈值（折内确定） | 阈值分析 | C2 Sensitivity: 0.047（默认）→ 0.512（nested）；Macro-F1: 0.459 → 0.674 | 默认阈值造成决策伪影；nested阈值恢复信号 | 不能把低Sensitivity等同于无信号 |

## 论文可直接使用的解释句

1. **患者语言信号**："C2 participant_speech的ROC-AUC为0.705，nested Sensitivity为0.512，但默认阈值下Sensitivity仅为0.047，表明是阈值伪影而非无信号。"

2. **症状证据片段信号**："C5 reviewed evidence的AUC为0.754，但domain_presence（AUC=0.790）和domain_count（AUC=0.794）表现相近，提示症状域覆盖驱动了大部分信号。"

3. **症状域覆盖信号**："domain_presence和domain_count的AUC达到0.790-0.794，高于C5 reviewed evidence，提示症状域覆盖是一个关键信号来源。"

4. **访谈协议结构**："C4 template_only（AUC=0.754）和template_presence（AUC=0.759）显示协议结构本身携带预测信号，独立于患者症状语言。"

5. **表征方式**："在MPNet下，participant_speech（C2）的AUC=0.764 > interviewer_speech（C3）的AUC=0.743；在BGE下，C3的AUC=0.769 > C2的AUC=0.704。"

6. **阈值策略**："C2 participant_speech在默认0.5阈值下Sensitivity接近0，但nested阈值恢复到0.512，说明分类表现依赖阈值策略。"

## 对应表格

- 主结果：`table_1_tfidf_primary.csv`
- 表征稳健性：`table_2_representation_robustness.csv`
- C5控制：`table_s4_c5_controls.csv`
- C4控制：`table_s6_c4_controls.csv`

## 方法说明

本分解表在所有主分析和控制分析完成后创建，不引入额外的事后模型筛选。仅作为解释框架，用于帮助读者组织理解结果。

## 引用格式

**表X. DAIC-WOZ PHQ-8阳性预测中的信号来源分解。** 识别六个信号层级：患者语言、症状证据片段、症状域覆盖、访谈协议结构、文本表征方式和阈值决策规则。每个层级提供代表性结果和解释边界。
