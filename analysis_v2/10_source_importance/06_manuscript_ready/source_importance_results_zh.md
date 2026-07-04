# 来源重要性分析结果

## 1. 增量模型表现

最优模型（M5: Patient + Domain + Protocol + C3 (sensitivity)）AUC=0.8356。

### 关键增量比较（5000次bootstrap CI）

- M1 vs M0（AUC）: +0.0741 [-0.0026, +0.1545] — 无显著改善
- M1 vs M0（Brier）: -0.0354 [-0.0632, -0.0050] — 显著改善
- M1 vs M0（LogLoss）: -0.0768 [-0.1410, -0.0064] — 显著改善
- M3 vs M0（AUC）: +0.1294 [+0.0460, +0.2138] — 显著改善
- M3 vs M0（Brier）: -0.0517 [-0.0809, -0.0198] — 显著改善
- M3 vs M0（LogLoss）: -0.1291 [-0.2012, -0.0495] — 显著改善
- M4 vs M3（AUC）: -0.0037 [-0.0203, +0.0131] — 无显著改善
- M4 vs M3（Brier）: +0.0025 [-0.0041, +0.0093] — 无显著改善
- M4 vs M3（LogLoss）: +0.0049 [-0.0107, +0.0215] — 无显著改善

## 2. 来源重要性排序

最重要的来源为**domain_count_prob**（ΔAUC=0.1810 ± 0.0542），其次为**template_presence_prob**（ΔAUC=0.1272 ± 0.0341）。**c2_prob** 在M3中无明显正向增量贡献（ΔAUC=-0.0054 ± 0.0140）。

## 3. Domain层级重要性

最重要的domain为mental_health_history_presence（删除后ΔAUC=+0.0594）。

## 4. 核心结论

1. 症状域覆盖（domain_count_prob）是综合模型中的主导信号来源
2. 患者语言（c2_prob）在控制domain和protocol后无独特贡献
3. C5 quote概率在domain_count进入后无增量价值（M4-M3 AUC Δ ≈ 0）
4. 临床背景域（functioning_impairment、mental_health_history）是domain层面的关键贡献者
