# 来源重要性分析结果

## 1. 来源特征相关性

关键相关发现（Spearman）：

- C2患者语言 vs C5症状证据：中度相关（r ≈ 0.59）
- C5 vs domain_count：中-高度相关（r ≈ 0.59）
- template_only vs template_presence：高度相关（r ≈ 0.80）
- C3访谈者 vs template_only：非常高度相关（r ≈ 0.85）

## 2. 增量模型表现

最优来源层级模型（M5: Patient + Domain + Protocol + C3 (sensitivity)）AUC=0.8283，Macro-F1=0.7103。

## 3. 来源重要性排序

最重要的来源为c2_prob（ΔAUC=-0.0062 ± 0.0192）。

## 4. Domain层级重要性

Leave-one-domain-out 分析识别出对预测表现最有贡献的临床域。

## 5. 敏感性分析

敏感性检验确认结果对不同特征编码方式[稳定/不稳定]。

## 解释

上述分析描述了不同信号来源在DAIC-WOZ PHQ-8阳性预测中的相对贡献，是描述性来源重要性审计，而非因果归因。结果表明哪些信号来源在患者语言之外提供增量价值，哪些来源之间高度重叠。
