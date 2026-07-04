# 访谈者偏差定位：E8/E9 机制分析摘要

生成时间：2026-06-16 16:31:00

## 分析目的

本分析对应实验序列中的 E8 和 E9。E8 提取 C3/C4/C5 的 top predictive n-grams，用于判断文本模型依赖哪些词项或访谈流程线索；E9 只使用访谈轮次、词数、诊断性访谈轮次数等计数特征建模，用于判断访谈者偏差信号是否可以被简单流程强度解释。

## E8：C3/C4/C5 top predictive n-grams

方法：对 `interviewer_only`、`interviewer_cleaned` 和 `symptom_evidence_only` 分别运行 5-fold TF-IDF + Logistic Regression / Linear SVM；每折只在训练折拟合 TF-IDF 和模型，汇总各 n-gram 的平均系数、出现折数和方向一致性。含下划线或字母数字混合的 prompt ID / annotation token 单独标记为 `transcript_prompt_id_or_annotation`，用于识别潜在转录结构 shortcut。下表先展示 TF-IDF + LR 的 positive-depression 前 15 个 n-grams。

### C3 interviewer_only：预测抑郁的访谈者侧 n-grams

| 排名 | n-gram         | 平均系数  | 出现折数 | 方向一致性 | 类别                          |
| -- | -------------- | ----- | ---- | ----- | --------------------------- |
| 1  | to therapy     | 0.483 | 5    | 1.000 | diagnostic_or_symptom_terms |
| 2  | therapy        | 0.475 | 5    | 1.000 | diagnostic_or_symptom_terms |
| 3  | they triggered | 0.425 | 5    | 1.000 | other_language_pattern      |
| 4  | they           | 0.425 | 5    | 1.000 | other_language_pattern      |
| 5  | triggered      | 0.425 | 5    | 1.000 | other_language_pattern      |
| 6  | triggered by   | 0.425 | 5    | 1.000 | other_language_pattern      |
| 7  | are they       | 0.425 | 5    | 1.000 | other_language_pattern      |
| 8  | by something   | 0.425 | 5    | 1.000 | other_language_pattern      |
| 9  | how long       | 0.414 | 5    | 1.000 | other_language_pattern      |
| 10 | long ago       | 0.414 | 5    | 1.000 | other_language_pattern      |
| 11 | ago were       | 0.414 | 5    | 1.000 | other_language_pattern      |
| 12 | you diagnosed  | 0.414 | 5    | 1.000 | diagnostic_or_symptom_terms |
| 13 | go to          | 0.411 | 5    | 1.000 | other_language_pattern      |
| 14 | still go       | 0.411 | 5    | 1.000 | other_language_pattern      |
| 15 | therapy now    | 0.411 | 5    | 1.000 | diagnostic_or_symptom_terms |

### C4 interviewer_cleaned：清洗后仍预测抑郁的访谈者侧 n-grams

| 排名 | n-gram                 | 平均系数  | 出现折数 | 方向一致性 | 类别                                 |
| -- | ---------------------- | ----- | ---- | ----- | ---------------------------------- |
| 1  | triggered by           | 0.501 | 5    | 1.000 | other_language_pattern             |
| 2  | they                   | 0.501 | 5    | 1.000 | other_language_pattern             |
| 3  | they triggered         | 0.501 | 5    | 1.000 | other_language_pattern             |
| 4  | triggered              | 0.501 | 5    | 1.000 | other_language_pattern             |
| 5  | are they               | 0.501 | 5    | 1.000 | other_language_pattern             |
| 6  | by something           | 0.501 | 5    | 1.000 | other_language_pattern             |
| 7  | sorry                  | 0.371 | 5    | 1.000 | backchannel_affiliative_response   |
| 8  | ellie17dec2012_03      | 0.361 | 5    | 1.000 | transcript_prompt_id_or_annotation |
| 9  | ellie17dec2012_03 that | 0.361 | 5    | 1.000 | transcript_prompt_id_or_annotation |
| 10 | trigger are            | 0.338 | 5    | 1.000 | other_language_pattern             |
| 11 | trigger                | 0.338 | 5    | 1.000 | other_language_pattern             |
| 12 | parent                 | 0.321 | 5    | 1.000 | other_language_pattern             |
| 13 | being                  | 0.313 | 5    | 1.000 | other_language_pattern             |
| 14 | mhm yeah               | 0.300 | 5    | 1.000 | backchannel_affiliative_response   |
| 15 | you stop               | 0.296 | 5    | 1.000 | other_language_pattern             |

### C5 symptom_evidence_only：预测抑郁的患者症状证据 n-grams

| 排名 | n-gram    | 平均系数  | 出现折数 | 方向一致性 | 类别                          |
| -- | --------- | ----- | ---- | ----- | --------------------------- |
| 1  | ago       | 0.516 | 5    | 1.000 | other_language_pattern      |
| 2  | couldn    | 0.464 | 5    | 1.000 | other_language_pattern      |
| 3  | all       | 0.399 | 5    | 1.000 | other_language_pattern      |
| 4  | years ago | 0.395 | 5    | 1.000 | other_language_pattern      |
| 5  | was       | 0.364 | 5    | 1.000 | other_language_pattern      |
| 6  | years     | 0.360 | 5    | 1.000 | other_language_pattern      |
| 7  | depressed | 0.342 | 5    | 1.000 | diagnostic_or_symptom_terms |
| 8  | two       | 0.338 | 5    | 1.000 | other_language_pattern      |
| 9  | and um    | 0.331 | 5    | 1.000 | other_language_pattern      |
| 10 | three     | 0.330 | 5    | 1.000 | other_language_pattern      |
| 11 | thinking  | 0.324 | 5    | 1.000 | other_language_pattern      |
| 12 | myself    | 0.324 | 5    | 1.000 | other_language_pattern      |
| 13 | some time | 0.319 | 5    | 1.000 | other_language_pattern      |
| 14 | just not  | 0.319 | 5    | 1.000 | other_language_pattern      |
| 15 | around    | 0.311 | 5    | 1.000 | other_language_pattern      |

### Top n-gram 类别分布

| 条件                  | 模型                  | 方向                      | 类别                                 | 数量 |
| ------------------- | ------------------- | ----------------------- | ---------------------------------- | -- |
| interviewer_cleaned | TF-IDF + LR         | negative_non_depression | other_language_pattern             | 27 |
| interviewer_cleaned | TF-IDF + LR         | negative_non_depression | transcript_prompt_id_or_annotation | 15 |
| interviewer_cleaned | TF-IDF + LR         | negative_non_depression | backchannel_affiliative_response   | 7  |
| interviewer_cleaned | TF-IDF + LR         | negative_non_depression | background_or_context_prompt       | 1  |
| interviewer_cleaned | TF-IDF + LR         | positive_depression     | other_language_pattern             | 38 |
| interviewer_cleaned | TF-IDF + LR         | positive_depression     | backchannel_affiliative_response   | 6  |
| interviewer_cleaned | TF-IDF + LR         | positive_depression     | transcript_prompt_id_or_annotation | 6  |
| interviewer_cleaned | TF-IDF + Linear SVM | negative_non_depression | other_language_pattern             | 24 |
| interviewer_cleaned | TF-IDF + Linear SVM | negative_non_depression | transcript_prompt_id_or_annotation | 14 |
| interviewer_cleaned | TF-IDF + Linear SVM | negative_non_depression | backchannel_affiliative_response   | 8  |
| interviewer_cleaned | TF-IDF + Linear SVM | negative_non_depression | background_or_context_prompt       | 4  |
| interviewer_cleaned | TF-IDF + Linear SVM | positive_depression     | other_language_pattern             | 24 |
| interviewer_cleaned | TF-IDF + Linear SVM | positive_depression     | transcript_prompt_id_or_annotation | 15 |
| interviewer_cleaned | TF-IDF + Linear SVM | positive_depression     | backchannel_affiliative_response   | 9  |
| interviewer_cleaned | TF-IDF + Linear SVM | positive_depression     | diagnostic_or_symptom_terms        | 1  |
| interviewer_cleaned | TF-IDF + Linear SVM | positive_depression     | interview_flow_or_followup         | 1  |
| interviewer_only    | TF-IDF + LR         | negative_non_depression | other_language_pattern             | 25 |
| interviewer_only    | TF-IDF + LR         | negative_non_depression | transcript_prompt_id_or_annotation | 16 |
| interviewer_only    | TF-IDF + LR         | negative_non_depression | backchannel_affiliative_response   | 6  |
| interviewer_only    | TF-IDF + LR         | negative_non_depression | background_or_context_prompt       | 2  |
| interviewer_only    | TF-IDF + LR         | negative_non_depression | diagnostic_or_symptom_terms        | 1  |
| interviewer_only    | TF-IDF + LR         | positive_depression     | other_language_pattern             | 31 |
| interviewer_only    | TF-IDF + LR         | positive_depression     | transcript_prompt_id_or_annotation | 11 |
| interviewer_only    | TF-IDF + LR         | positive_depression     | backchannel_affiliative_response   | 4  |
| interviewer_only    | TF-IDF + LR         | positive_depression     | diagnostic_or_symptom_terms        | 4  |
| interviewer_only    | TF-IDF + Linear SVM | negative_non_depression | other_language_pattern             | 23 |
| interviewer_only    | TF-IDF + Linear SVM | negative_non_depression | transcript_prompt_id_or_annotation | 12 |
| interviewer_only    | TF-IDF + Linear SVM | negative_non_depression | backchannel_affiliative_response   | 8  |
| interviewer_only    | TF-IDF + Linear SVM | negative_non_depression | background_or_context_prompt       | 4  |
| interviewer_only    | TF-IDF + Linear SVM | negative_non_depression | diagnostic_or_symptom_terms        | 2  |

## E9：访谈者计数特征 baseline

方法：只使用轮次数、词数、诊断性访谈轮次数、清洗后访谈者词数等数值特征，运行 StandardScaler + Logistic Regression(class_weight='balanced') 的 5-fold CV。该分析不使用任何文本词项。

| 特征集                           | Macro-F1 | MCC   | ROC-AUC | PR-AUC | 阳性F1  | Recall | Precision |
| ----------------------------- | -------- | ----- | ------- | ------ | ----- | ------ | --------- |
| cleaned_interviewer_counts    | 0.545    | 0.132 | 0.556   | 0.295  | 0.373 | 0.500  | 0.304     |
| interviewer_process_counts    | 0.533    | 0.127 | 0.590   | 0.332  | 0.358 | 0.525  | 0.276     |
| diagnostic_only_counts        | 0.543    | 0.125 | 0.559   | 0.344  | 0.348 | 0.478  | 0.280     |
| all_transcript_count_controls | 0.487    | 0.032 | 0.564   | 0.315  | 0.299 | 0.450  | 0.228     |

### 计数特征组间差异

| 特征                                    | 非抑郁均值    | 抑郁均值     | 差值       | Mann-Whitney p | BH q  | PHQ-8 rho |
| ------------------------------------- | -------- | -------- | -------- | -------------- | ----- | --------- |
| diagnostic_interviewer_turns          | 5.136    | 5.881    | 0.745    | 0.028          | 0.170 | 0.343     |
| diagnostic_turns_removed              | 5.136    | 5.881    | 0.745    | 0.028          | 0.170 | 0.343     |
| interviewer_cleaned_word_ratio        | 0.900    | 0.909    | 0.009    | 0.049          | 0.170 | -0.311    |
| diagnostic_turn_ratio                 | 0.065    | 0.072    | 0.006    | 0.052          | 0.170 | 0.271     |
| interviewer_turns                     | 78.224   | 83.333   | 5.109    | 0.072          | 0.186 | 0.209     |
| interviewer_cleaned_turns             | 73.088   | 77.452   | 4.364    | 0.114          | 0.248 | 0.167     |
| participant_words                     | 1496.973 | 1380.952 | -116.020 | 0.246          | 0.456 | -0.044    |
| interviewer_word_share                | 0.304    | 0.326    | 0.021    | 0.293          | 0.477 | 0.048     |
| interviewer_cleaned_words             | 498.986  | 501.595  | 2.609    | 0.376          | 0.543 | -0.060    |
| interviewer_words                     | 543.020  | 551.381  | 8.361    | 0.493          | 0.611 | -0.017    |
| participant_to_interviewer_word_ratio | 2.754    | 2.624    | -0.130   | 0.517          | 0.611 | -0.022    |
| participant_turns                     | 171.102  | 172.595  | 1.493    | 0.792          | 0.858 | 0.061     |

### 计数模型中绝对系数较大的特征

| 特征集                           | 特征                                    | 标准化平均系数 | 方向                      | 折数 |
| ----------------------------- | ------------------------------------- | ------- | ----------------------- | -- |
| all_transcript_count_controls | participant_words                     | -0.740  | negative_non_depression | 5  |
| all_transcript_count_controls | participant_to_interviewer_word_ratio | 0.423   | positive_depression     | 5  |
| all_transcript_count_controls | turns_total                           | 0.277   | positive_depression     | 5  |
| all_transcript_count_controls | interviewer_cleaned_words             | -0.264  | negative_non_depression | 5  |
| all_transcript_count_controls | interviewer_word_share                | 0.263   | positive_depression     | 5  |
| all_transcript_count_controls | participant_turns                     | 0.244   | positive_depression     | 5  |
| all_transcript_count_controls | interviewer_turns                     | 0.182   | positive_depression     | 5  |
| all_transcript_count_controls | interviewer_cleaned_turns             | 0.174   | positive_depression     | 5  |
| all_transcript_count_controls | diagnostic_interviewer_turns          | 0.148   | positive_depression     | 5  |
| all_transcript_count_controls | diagnostic_turns_removed              | 0.148   | positive_depression     | 5  |
| all_transcript_count_controls | diagnostic_turn_ratio                 | 0.136   | positive_depression     | 5  |
| all_transcript_count_controls | interviewer_words                     | -0.128  | negative_non_depression | 5  |
| cleaned_interviewer_counts    | interviewer_cleaned_turns             | 0.392   | positive_depression     | 5  |
| cleaned_interviewer_counts    | interviewer_cleaned_words             | -0.174  | negative_non_depression | 5  |
| cleaned_interviewer_counts    | interviewer_cleaned_word_ratio        | 0.017   | positive_depression     | 5  |
| diagnostic_only_counts        | diagnostic_interviewer_turns          | 0.264   | positive_depression     | 5  |
| diagnostic_only_counts        | diagnostic_turns_removed              | 0.264   | positive_depression     | 5  |
| diagnostic_only_counts        | diagnostic_turn_ratio                 | -0.141  | negative_non_depression | 5  |
| interviewer_process_counts    | interviewer_cleaned_words             | -0.228  | negative_non_depression | 5  |
| interviewer_process_counts    | interviewer_turns                     | 0.195   | positive_depression     | 5  |

## 当前解释

1. 如果 C3/C4 的 positive n-grams 主要集中在诊断性或追问流程词项，说明访谈者侧性能并不等同于患者自发表达，而可能利用了访谈流程中的互动线索。
2. 如果 C4 清洗后仍出现 transcript_prompt_id_or_annotation 类 n-grams，说明当前 C3/C4 文本中还保留了 DAIC-WOZ prompt ID 或转录标注残留；这不是临床语言本身，但恰好说明访谈者侧输入容易携带结构性 shortcut。
3. 如果 C4 清洗后仍出现较多 interview_flow_or_followup 或 background_or_context_prompt 类 n-grams，说明显性症状问题被删除后，访谈结构和动态追问仍可能保留预测信息。
4. 如果 E9 计数特征也有非零 MCC/PR-AUC，说明偏差信号至少部分来自访谈强度或诊断性追问数量；如果 E9 明显弱于 C3/C4 文本模型，则说明具体措辞、prompt ID/annotation 和语义模式仍然重要。
5. 该发现提示后续应增加一个 prompt-ID-stripped 敏感性分析：删除 `happy_lasttime`、`bouts_symptoms`、`Ellie17Dec2012_03` 等结构化标记后，重跑 C3/C4 文本模型，检验访谈者偏差信号是否仍然存在。

## 输出文件

- `B20_E8_top_predictive_ngrams_top50.csv`：E8 top n-grams 摘要。
- `B21_E8_top_predictive_ngrams_long.csv`：E8 每折 n-gram 系数长表。
- `B22_E8_top_ngram_category_summary.csv`：E8 top n-gram 类别分布。
- `B23_E9访谈者计数特征baseline_五折CV汇总.csv`：E9 计数特征模型汇总。
- `B24_E9访谈者计数特征baseline_逐折明细.csv`：E9 逐折结果。
- `B25_E9访谈者计数特征baseline_系数.csv`：E9 标准化系数。
- `B26_E9访谈者计数特征_组间差异.csv`：计数特征的组间差异和 PHQ-8 相关。