# Targeted literature search log

## Scope

This search was added after the submission audit to constrain the novelty wording. It is a targeted, reproducible search of directly related work, not a systematic review or meta-analysis（非系统综述）.

- Search date: 2026-07-12 (Asia/Shanghai)
- 检索截止日期: 2026-07-12
- Databases and sources / 数据库与来源: ACL Anthology, ISCA Archive, and PubMed. The web search layer was used only to locate records on those primary bibliographic or publisher pages.
- Dataset scope: DAIC-WOZ or E-DAIC interview text.
- Topic scope: speaker/source separation, interviewer prompts or protocol structure, text-budget controls, symptom/expert interpretation, calibration or uncertainty, and evidence traceability.

## Queries

The following query families were run with dataset names combined with the audit concepts. Equivalent capitalization and singular/plural variants were allowed.

1. `DAIC-WOZ interviewer participant prompt bias depression`
2. `DAIC-WOZ depression interview transcript calibration threshold`
3. `DAIC-WOZ symptom depression evidence expert annotation`
4. `E-DAIC DAIC-WOZ interpretable depression large language model`
5. `DAIC-WOZ interviewer bias adversarial contextual positional encoding`
6. `DAIC-WOZ depression textual time series calibrated uncertainty`

Domain restrictions were applied separately to `aclanthology.org`, `isca-archive.org`, and `pubmed.ncbi.nlm.nih.gov`. Citation metadata and abstracts were verified on the primary record page. No claim in the manuscript depends on a search-result snippet alone.

## Eligibility

### 纳入标准

- Peer-reviewed conference or journal article available by the cutoff date.
- Uses DAIC-WOZ or E-DAIC interview text as an analyzed input.
- Directly addresses at least one audit dimension: speaker/source handling, prompts/protocol/interaction, text quantity or position, symptom/expert representation, calibration/uncertainty, or source-grounded evidence.
- Sufficient primary metadata was available to verify title, authors, year, venue, and DOI or stable record URL.

### 排除标准

- Acoustic-only or visual-only work with no transcript analysis.
- General social-media depression detection without DAIC-WOZ/E-DAIC interviews.
- Reviews, theses, non-peer-reviewed summaries, duplicate records, or papers cited only for generic algorithms.
- Papers that mention DAIC-WOZ only as background and do not report an analysis on it.

## Retained representative studies

| Study | Primary record | Audit relevance |
|---|---|---|
| López-Otero et al. (2017) | ISCA Archive, doi:10.21437/Interspeech.2017-1201 | Participant transcription; calibration/operating point |
| Mallol-Ragolta et al. (2019) | ISCA Archive, doi:10.21437/Interspeech.2019-2036 | Hierarchical transcript modeling |
| Rinaldi et al. (2020) | ACL Anthology, doi:10.18653/v1/2020.acl-main.2 | Latent categorization of interview prompts |
| Burdisso et al. (2024) | ACL Anthology, doi:10.18653/v1/2024.clinicalnlp-1.8 | Participant versus interviewer prompts and prompt regions |
| Agarwal et al. (2024) | ACL Anthology, 2024.lrec-main.87 | Psychiatric expert symptom annotations |
| Zhao et al. (2025) | ACL Anthology, doi:10.18653/v1/2025.findings-acl.1181 | Theme and interaction modeling |
| Zhang and Poellabauer (2025) | ACL Anthology, doi:10.18653/v1/2025.findings-emnlp.650 | Question-context invariance and interviewer-bias mitigation |
| Lee et al. (2026) | PubMed/PLOS, doi:10.1371/journal.pdig.0001205 | LLM-extracted interpretable factors; DAIC-WOZ/E-DAIC |
| Schmidt et al. (2026) | ACL Anthology, doi:10.18653/v1/2026.findings-acl.1630 | Probabilistic text time series and calibrated uncertainty |

## Permitted novelty wording

The search does not support a field-wide claim of novelty or rarity. The permitted statement is:

> Among the representative studies retained by this targeted search, we did not identify one analysis that jointly integrated source separation, protocol/structure controls, symmetric text-budget sensitivity, fold-internal calibration and thresholding, and retained-evidence traceability.

The manuscript must not use “first”, “the literature rarely”, or an equivalent universal claim on the basis of this log.
