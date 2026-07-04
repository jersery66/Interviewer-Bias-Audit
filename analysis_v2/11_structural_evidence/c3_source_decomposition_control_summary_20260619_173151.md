# C3来源分解控制分析摘要

运行时间：20260619_173151

## 1. 长度/词数控制

为检验 C4_retained 的性能是否主要由文本长度驱动，本分析从每名被试的 C4_retained 中随机抽取连续 word window，构造两个长度控制条件：固定 45 词版本，以及按被试层面 C3_removed 词数匹配版本。每个条件重复随机抽样 30 次；每次抽样后采用与主实验一致的 TF-IDF + Logistic Regression 和 10×5 repeated CV。

| 条件 | 平均词数 | Macro-F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| C4_retained downsampled to 45 words | 44.3 | 0.537 ± 0.040 | 0.092 ± 0.080 | 0.570 ± 0.053 |
| C4_retained word-count matched to C3_removed | 45.3 | 0.547 ± 0.041 | 0.112 ± 0.082 | 0.583 ± 0.062 |

### 长度控制 OOF 配对比较

| 比较 | 指标 | A | B | 差值(A-B) | 95% CI | p |
|---|---|---:|---:|---:|---|---:|
| C4_fixed45 vs C2 participant-only | macro_f1 | 0.580 | 0.462 | +0.118 | [+0.012, +0.217] | 0.0530 |
| C4_fixed45 vs C2 participant-only | mcc | 0.202 | 0.136 | +0.066 | [-0.148, +0.347] | 0.6755 |
| C4_fixed45 vs C2 participant-only | auc | 0.695 | 0.763 | -0.069 | [-0.166, +0.024] | 0.1430 |
| C4_matched_to_C3_removed vs C2 participant-only | macro_f1 | 0.619 | 0.462 | +0.157 | [+0.075, +0.243] | 0.0065 |
| C4_matched_to_C3_removed vs C2 participant-only | mcc | 0.287 | 0.136 | +0.150 | [-0.021, +0.393] | 0.2515 |
| C4_matched_to_C3_removed vs C2 participant-only | auc | 0.725 | 0.763 | -0.039 | [-0.116, +0.045] | 0.3845 |
| C4_matched_to_C3_removed vs C3_removed | macro_f1 | 0.619 | 0.700 | -0.081 | [-0.182, +0.018] | 0.1450 |
| C4_matched_to_C3_removed vs C3_removed | mcc | 0.287 | 0.441 | -0.154 | [-0.358, +0.018] | 0.1705 |
| C4_matched_to_C3_removed vs C3_removed | auc | 0.725 | 0.810 | -0.085 | [-0.174, -0.002] | 0.0995 |
| C4_fixed45 vs original C4_retained | macro_f1 | 0.580 | 0.719 | -0.140 | [-0.231, -0.050] | 0.0095 |
| C4_fixed45 vs original C4_retained | mcc | 0.202 | 0.453 | -0.251 | [-0.435, -0.082] | 0.0160 |
| C4_fixed45 vs original C4_retained | auc | 0.695 | 0.815 | -0.120 | [-0.171, -0.068] | 0.0000 |
| C4_matched_to_C3_removed vs original C4_retained | macro_f1 | 0.619 | 0.719 | -0.101 | [-0.200, -0.013] | 0.0595 |
| C4_matched_to_C3_removed vs original C4_retained | mcc | 0.287 | 0.453 | -0.166 | [-0.344, -0.004] | 0.1185 |
| C4_matched_to_C3_removed vs original C4_retained | auc | 0.725 | 0.815 | -0.090 | [-0.139, -0.038] | 0.0080 |

## 2. C2 与 C4_retained 概率相关

C2 participant-only 与 C4_retained 的 OOF 概率 Spearman 相关为 rho=0.589（p=4.862e-19），Pearson r=0.603（p=4.591e-20）。C4_retained 正确而 C2 错误的样本为 28/189（占全样本 0.148；占 C2 错误样本 0.683）。

## 3. Top n-grams 初步类别审计

以下为 top n-grams 的规则化人工归类草稿，用于提示后续人工复核重点。

| 条件 | 方向 | 类别 | n |
|---|---|---|---:|
| c3_full | negative_for_PHQ8_positive | 非特异噪声/待人工复核 | 25 |
| c3_full | negative_for_PHQ8_positive | 中性反馈/backchannel | 10 |
| c3_full | negative_for_PHQ8_positive | 显性心理健康/症状提问 | 5 |
| c3_full | negative_for_PHQ8_positive | 常规背景/生活问题 | 4 |
| c3_full | negative_for_PHQ8_positive | 转录标记/噪声 | 4 |
| c3_full | negative_for_PHQ8_positive | 共情回应 | 2 |
| c3_full | positive_for_PHQ8_positive | 非特异噪声/待人工复核 | 28 |
| c3_full | positive_for_PHQ8_positive | 显性心理健康/症状提问 | 9 |
| c3_full | positive_for_PHQ8_positive | 转录标记/噪声 | 4 |
| c3_full | positive_for_PHQ8_positive | 追问触发 | 4 |
| c3_full | positive_for_PHQ8_positive | 共情回应 | 3 |
| c3_full | positive_for_PHQ8_positive | 中性反馈/backchannel | 2 |
| c3_removed | negative_for_PHQ8_positive | 显性心理健康/症状提问 | 25 |
| c3_removed | negative_for_PHQ8_positive | 非特异噪声/待人工复核 | 18 |
| c3_removed | negative_for_PHQ8_positive | 常规背景/生活问题 | 5 |
| c3_removed | negative_for_PHQ8_positive | 转录标记/噪声 | 2 |
| c3_removed | positive_for_PHQ8_positive | 非特异噪声/待人工复核 | 33 |
| c3_removed | positive_for_PHQ8_positive | 显性心理健康/症状提问 | 16 |
| c3_removed | positive_for_PHQ8_positive | 追问触发 | 1 |
| c4_retained | negative_for_PHQ8_positive | 非特异噪声/待人工复核 | 26 |
| c4_retained | negative_for_PHQ8_positive | 中性反馈/backchannel | 12 |
| c4_retained | negative_for_PHQ8_positive | 常规背景/生活问题 | 5 |
| c4_retained | negative_for_PHQ8_positive | 转录标记/噪声 | 5 |
| c4_retained | negative_for_PHQ8_positive | 共情回应 | 2 |
| c4_retained | positive_for_PHQ8_positive | 非特异噪声/待人工复核 | 33 |
| c4_retained | positive_for_PHQ8_positive | 转录标记/噪声 | 5 |
| c4_retained | positive_for_PHQ8_positive | 追问触发 | 5 |
| c4_retained | positive_for_PHQ8_positive | 共情回应 | 4 |
| c4_retained | positive_for_PHQ8_positive | 中性反馈/backchannel | 3 |

## 4. 空文本敏感性

删除 3 个访谈者侧空文本样本后重新运行 10×5 repeated CV，结果如下。

| 条件 | Macro-F1 | MCC | AUC |
|---|---:|---:|---:|
| C2 participant-only reference | 0.471 ± 0.061 | 0.080 ± 0.147 | 0.748 ± 0.105 |
| C3_full / interviewer-only text | 0.724 ± 0.078 | 0.473 ± 0.159 | 0.841 ± 0.068 |
| C3_removed / removed explicit clinical prompts | 0.708 ± 0.041 | 0.458 ± 0.081 | 0.829 ± 0.062 |
| C4_retained / retained non-explicit interviewer text | 0.696 ± 0.085 | 0.419 ± 0.172 | 0.820 ± 0.079 |

## 初步解释

若长度匹配后的 C4_retained 仍保持高于 C2 的表现，可说明非显性访谈者话语的标签相关信号不只是文本更长造成的。若明显下降，则说明 C4_retained 的完整性能部分依赖文本量。无论哪种情况，该分析均应被表述为敏感性/机制分析，而不是严格因果分解。