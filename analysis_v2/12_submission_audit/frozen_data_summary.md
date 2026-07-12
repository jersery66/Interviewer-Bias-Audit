# 冻结结果数据汇总

## 0. 计算字段

| 字段 | 数值 |
|---|---:|
| n | 142 |
| n_positive | 43 |
| n_negative | 99 |
| PHQ-8_cutoff | 10 |
| participant_level_folds | 5 |
| bootstrap_resamples | 5000 |
| paired_permutations | 10000 |
| token_draws | 50 |
| base_seed | 20260705 |
| half_min_permutation_seed | 20765706 |

\[
\mathrm{BSS}=1-\frac{\mathrm{Brier}_{model}}{\mathrm{Brier}_{null}}
\]

## 1. 单来源指标

| condition | n | n_positive | ROC AUC | AUC CI low | AUC CI high | PR-AUC | raw Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| participant_speech | 142 | 43 | 0.7172 | 0.6140 | 0.8111 | 0.6506 | 0.2307 |
| interviewer_speech | 142 | 43 | 0.8081 | 0.7287 | 0.8790 | 0.6610 | 0.1959 |
| full_transcript | 142 | 43 | 0.7602 | 0.6702 | 0.8436 | 0.5737 | 0.2174 |
| non_explicit_interviewer_speech | 142 | 43 | 0.7815 | 0.6965 | 0.8562 | 0.6427 | 0.2034 |
| participant_symptom_evidence | 142 | 43 | 0.7353 | 0.6397 | 0.8187 | 0.5583 | 0.2245 |

| condition | raw ECE 5-bin | raw calibration intercept | raw calibration slope | macro F1 @0.50 | sensitivity @0.50 | specificity @0.50 | macro F1 nested | sensitivity nested | specificity nested |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| participant_speech | 0.2276 | 1.0110 | 12.8357 | 0.4587 | 0.0465 | 1.0000 | 0.6882 | 0.4884 | 0.8687 |
| interviewer_speech | 0.1572 | -0.6502 | 2.7271 | 0.7297 | 0.6977 | 0.7879 | 0.7138 | 0.6744 | 0.7778 |
| full_transcript | 0.1759 | -0.1920 | 4.8548 | 0.6526 | 0.4419 | 0.8485 | 0.6441 | 0.5814 | 0.7273 |
| non_explicit_interviewer_speech | 0.1591 | -0.6254 | 2.6243 | 0.7072 | 0.6279 | 0.7980 | 0.6995 | 0.5116 | 0.8687 |
| participant_symptom_evidence | 0.1763 | -0.3488 | 5.2963 | 0.6742 | 0.5116 | 0.8283 | 0.6788 | 0.5349 | 0.8182 |

## 2. Brier 与 BSS

| condition | probability_variant | model Brier | Brier CI low | Brier CI high | trainfold null Brier | BSS trainfold | BSS CI low | BSS CI high | cohort prevalence | cohort-null BSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| participant_speech | raw | 0.2307 | 0.2279 | 0.2334 | 0.2112 | -0.0921 | -0.1048 | -0.0795 | 0.3028 | -0.0927 |
| participant_speech | platt | 0.1792 | 0.1586 | 0.2012 | 0.2112 | 0.1517 | 0.0472 | 0.2486 | 0.3028 | 0.1513 |
| participant_speech | isotonic | 0.1764 | 0.1475 | 0.2060 | 0.2112 | 0.1649 | 0.0256 | 0.3008 | 0.3028 | 0.1645 |
| interviewer_speech | raw | 0.1959 | 0.1805 | 0.2114 | 0.2112 | 0.0725 | -0.0012 | 0.1454 | 0.3028 | 0.0720 |
| interviewer_speech | platt | 0.1593 | 0.1293 | 0.1914 | 0.2112 | 0.2458 | 0.0933 | 0.3864 | 0.3028 | 0.2454 |
| interviewer_speech | isotonic | 0.1718 | 0.1377 | 0.2091 | 0.2112 | 0.1867 | 0.0098 | 0.3478 | 0.3028 | 0.1863 |
| full_transcript | raw | 0.2174 | 0.2099 | 0.2250 | 0.2112 | -0.0293 | -0.0648 | 0.0066 | 0.3028 | -0.0298 |
| full_transcript | platt | 0.1752 | 0.1506 | 0.2010 | 0.2112 | 0.1706 | 0.0485 | 0.2866 | 0.3028 | 0.1701 |
| full_transcript | isotonic | 0.1836 | 0.1501 | 0.2199 | 0.2112 | 0.1308 | -0.0410 | 0.2890 | 0.3028 | 0.1304 |
| non_explicit_interviewer_speech | raw | 0.2034 | 0.1895 | 0.2179 | 0.2112 | 0.0372 | -0.0313 | 0.1026 | 0.3028 | 0.0367 |
| non_explicit_interviewer_speech | platt | 0.1680 | 0.1435 | 0.1945 | 0.2112 | 0.2048 | 0.0791 | 0.3208 | 0.3028 | 0.2044 |
| non_explicit_interviewer_speech | isotonic | 0.1656 | 0.1379 | 0.1955 | 0.2112 | 0.2158 | 0.0744 | 0.3466 | 0.3028 | 0.2154 |
| participant_symptom_evidence | raw | 0.2245 | 0.2178 | 0.2311 | 0.2112 | -0.0626 | -0.0934 | -0.0313 | 0.3028 | -0.0632 |
| participant_symptom_evidence | platt | 0.1865 | 0.1620 | 0.2127 | 0.2112 | 0.1172 | -0.0071 | 0.2323 | 0.3028 | 0.1167 |
| participant_symptom_evidence | isotonic | 0.1961 | 0.1629 | 0.2314 | 0.2112 | 0.0715 | -0.0948 | 0.2292 | 0.3028 | 0.0710 |

## 3. 配对比较

| comparison | delta AUC | CI low | CI high | raw p | q | n_bootstrap | n_permutations |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_transcript vs participant_speech | 0.0430 | -0.0395 | 0.1301 | 0.3611 | 0.4731 | 5000 | 10000 |
| interviewer_speech vs participant_speech | 0.0909 | -0.0004 | 0.1898 | 0.0876 | 0.4380 | 5000 | 10000 |
| participant_symptom_evidence vs participant_speech | 0.0181 | -0.0790 | 0.1069 | 0.7273 | 0.7273 | 5000 | 10000 |
| M5 vs M3 | 0.0028 | -0.0040 | 0.0101 | 0.3785 | 0.4731 | 5000 | 10000 |
| M4 vs M3 | 0.0035 | -0.0038 | 0.0114 | 0.2499 | 0.4731 | 5000 | 10000 |

## 4. 结构模型

| model | ROC AUC | PR-AUC | Brier | log loss |
|---|---:|---:|---:|---:|
| S_length | 0.5694 | 0.3719 | 0.2466 | 0.6882 |
| S_interaction | 0.5729 | 0.3707 | 0.2470 | 0.6876 |
| S_protocol | 0.6754 | 0.5026 | 0.2245 | 0.6428 |
| S_all | 0.7289 | 0.5219 | 0.2101 | 0.6149 |

## 5. 文本长度与窗口

### 5.1 原等量控制审计

| source | original token Q1 | original token median | original token Q3 | truncated n | mean retained fraction | retained fraction Q1 | retained fraction median | retained fraction Q3 | zero source n | target token median | zero target n | target below 10 n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| participant | 850.0 | 1243.5 | 1744.5 | 132 | 0.4590 | 0.2614 | 0.3983 | 0.5905 | 0 | 501.0 | 2 | 2 |
| interviewer | 453.8 | 503.5 | 540.8 | 10 | 0.9873 | 1.0000 | 1.0000 | 1.0000 | 2 | 501.0 | 2 | 2 |
| shared_target | — | — | — | — | — | — | — | — | 0 | 501.0 | 2 | 2 |

### 5.2 原等量控制的模型指标

| condition | n_draws | mean AUC | SD AUC | draw AUC Q2.5 | draw AUC Q97.5 | joint CI low | joint CI high | mean PR-AUC | mean Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| participant_matched | 50 | 0.6508 | 0.0461 | 0.5781 | 0.7275 | 0.5135 | 0.7801 | 0.4919 | 0.2311 |
| interviewer_matched | 50 | 0.8084 | 0.0016 | 0.8055 | 0.8113 | 0.7259 | 0.8821 | 0.6628 | 0.1964 |

| comparison | mean delta AUC | SD delta AUC | draw delta Q2.5 | draw delta Q97.5 | joint CI low | joint CI high | raw p | q | n_permutations | permutation seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interviewer_matched vs participant_matched | 0.1576 | 0.0462 | 0.0811 | 0.2315 | 0.0225 | 0.2939 | 0.0019 | 0.0019 | 10000 | 20765706 |

### 5.3 对称半最小预算

| condition | n_draws | mean AUC | SD AUC | draw AUC Q2.5 | draw AUC Q97.5 | joint CI low | joint CI high | mean PR-AUC | mean Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| participant_matched | 50 | 0.5883 | 0.0583 | 0.4855 | 0.6818 | 0.4402 | 0.7367 | 0.4124 | 0.2329 |
| interviewer_matched | 50 | 0.7057 | 0.0462 | 0.6124 | 0.7765 | 0.5633 | 0.8238 | 0.5332 | 0.2141 |

| comparison | mean delta AUC | SD delta AUC | draw delta Q2.5 | draw delta Q97.5 | joint CI low | joint CI high | raw p | q | n_permutations | permutation seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interviewer_matched vs participant_matched | 0.1174 | 0.0693 | -0.0134 | 0.2320 | -0.0691 | 0.2953 | 0.0039 | 0.0039 | 10000 | 20765706 |

### 5.4 对称位置

| position | model | ROC AUC | PR-AUC | Brier | log loss | AUC CI low | AUC CI high |
|---|---|---:|---:|---:|---:|---:|---:|
| early | participant_matched | 0.4346 | 0.2953 | 0.2397 | 0.6725 | 0.3313 | 0.5431 |
| early | interviewer_matched | 0.6984 | 0.4714 | 0.2221 | 0.6363 | 0.6037 | 0.7833 |
| middle | participant_matched | 0.7625 | 0.5908 | 0.2225 | 0.6378 | 0.6737 | 0.8434 |
| middle | interviewer_matched | 0.7381 | 0.5845 | 0.2052 | 0.6011 | 0.6398 | 0.8241 |
| late | participant_matched | 0.7256 | 0.5723 | 0.2283 | 0.6495 | 0.6299 | 0.8204 |
| late | interviewer_matched | 0.8184 | 0.6490 | 0.1869 | 0.5611 | 0.7418 | 0.8834 |

| position | delta AUC | CI low | CI high | raw p | q | n_permutations |
|---|---:|---:|---:|---:|---:|---:|
| early | 0.2638 | 0.1440 | 0.3794 | 0.0001 | 0.0003 | 10000 |
| middle | -0.0244 | -0.1259 | 0.0756 | 0.6693 | 0.6693 | 10000 |
| late | 0.0928 | -0.0246 | 0.2100 | 0.1196 | 0.1794 | 10000 |

## 6. 保留片段来源对齐

| field | n |
|---|---:|
| candidate_span_n | 1143 |
| final_span_n | 1137 |
| participant_n_with_evidence | 141 |
| participant_n_total | 142 |
| unchanged_source_matched_n | 1102 |
| manually_corrected_to_source_n | 35 |
| excluded_duplicate_n | 5 |
| excluded_invalid_n | 1 |
| manual_audit_row_n | 41 |
| final_source_match_n | 1137 |
| unique_location_n | 1136 |
| multiple_exact_location_n | 1 |
| not_found_after_review_n | 0 |
| changed_participant_n | 24 |
| token_count_delta_whitespace | 103 |

| representation | ROC AUC | PR-AUC | Brier | log loss | AUC CI low | AUC CI high |
|---|---:|---:|---:|---:|---:|---:|
| original_retained | 0.7395 | 0.5571 | 0.2242 | 0.6410 | 0.6528 | 0.8231 |
| aligned_retained | 0.7353 | 0.5583 | 0.2245 | 0.6415 | 0.6424 | 0.8198 |

| metric | aligned minus original | CI low | CI high | raw p | q | n_permutations |
|---|---:|---:|---:|---:|---:|---:|
| roc_auc | -0.0042 | -0.0148 | 0.0049 | 0.3160 | 0.4740 | 10000 |
| pr_auc | 0.0011 | -0.0147 | 0.0173 | 0.8573 | 0.8573 | 10000 |
| brier | 0.0003 | -0.0002 | 0.0008 | 0.2714 | 0.4740 | 10000 |

| probability absolute change | value |
|---|---:|
| n | 142 |
| mean | 0.0018 |
| median | 0.0012 |
| Q1 | 0.0005 |
| Q3 | 0.0020 |
| Q95 | 0.0048 |
| max | 0.0158 |
| n_ge_0.01 | 3 |
| n_ge_0.05 | 0 |

## 7. E-DAIC

| field | n |
|---|---:|
| participants | 22 |
| positive | 6 |

## 8. 复算与完整性

| field | value |
|---|---:|
| recomputation_status | PASS |
| recomputation_file_count | 43 |
| recomputation_all_equal | true |
| output_manifest_entries | 71 |
| local_pytest_passed | 98 |
| github_actions_run_id | 29191567287 |
| github_actions_status | success |
| git_commit | 9ad520808ff71ba81d59ac458766eff3592d3a30 |
