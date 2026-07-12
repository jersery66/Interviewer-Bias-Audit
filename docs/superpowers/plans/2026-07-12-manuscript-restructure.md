# Manuscript Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the frozen DAIC-WOZ audit outputs while creating a clean, reader-facing Chinese manuscript whose single narrative is the source decomposition of PHQ-8 predictive information in semi-structured interviews.

**Architecture:** Keep `manuscript_submission_zh.md` as the technical audit draft and create a separate `main_text/manuscript_restructured_zh.md` as the submission-oriented paper. The new manuscript uses five reader-facing sections (摘要、引言、方法、结果、讨论) plus declarations and references; internal model IDs, audit timeline language, and result-checking detail remain in supplements and audit files. Tests read the frozen CSV/JSON outputs and assert that the new prose contains only supported numbers and permitted terminology.

**Tech Stack:** UTF-8 Markdown, Python 3.12, pytest, pandas, JSON/CSV frozen outputs.

---

### Task 1: Lock the manuscript contract with a failing test

**Files:**
- Create: `tests/test_restructured_manuscript.py`
- Create: `docs/superpowers/plans/2026-07-12-manuscript-restructure.md`

- [x] **Step 1: Write tests** for the new manuscript path, five-part reader-facing structure, three results subsections, five discussion paragraphs, absence of RQ/M/C5/internal audit labels in reader-facing prose, and alignment of every headline number with frozen outputs.
- [x] **Step 2: Run the focused test** with `E:\Anaconda\Scripts\pytest.exe -q tests/test_restructured_manuscript.py` and verify it fails because the new manuscript does not yet exist.

### Task 2: Draft the clean Chinese manuscript from the frozen evidence

**Files:**
- Create: `analysis_v2/06_tables_figures/main_text/manuscript_restructured_zh.md`

- [x] **Step 1:** Write the recommended title and a one-paragraph abstract containing only sample/positive count, four source AUCs, the 0.0028 conditional increment, and the symmetric-budget interval.
- [x] **Step 2:** Write four natural Introduction paragraphs covering background, interpretation gap, predictive sufficiency versus conditional increment, and study aim; move the literature-design comparison out of the main narrative.
- [x] **Step 3:** Write four Method subsections: data/outcome; source and variable construction; model/evaluation; comparison and sensitivity analysis. Define the training-fold Brier null, symmetric half-min budget, position windows, C5 retained-span boundary, and cross-fitting without exposing internal audit identifiers.
- [x] **Step 4:** Write three Results subsections: source performance; interviewer-language conditional increment and structural information; text budget/window/C5 sensitivity. Use the frozen numeric outputs and keep calibration, BSS, and C5 traceability at supporting-result density.
- [x] **Step 5:** Write one five-paragraph Discussion with central finding, interpretation of interviewer-side information, budget/structure meaning, calibration/C5 methodological boundary, and limitations/conclusion.
- [x] **Step 6:** Add data/code availability, ethics, abbreviations/statistical note, supplement map, and references without claiming unrestricted full data-to-result reproduction.

### Task 3: Update the authoritative entry points and consistency checks

**Files:**
- Modify: `analysis_v2/README.md`
- Modify: `tests/test_restructured_manuscript.py`
- Modify: `analysis_v2/06_tables_figures/manuscript_review_and_revision_log.md`

- [x] **Step 1:** Point the analysis README to the new clean manuscript and label the existing submission draft as the technical audit底稿.
- [x] **Step 2:** Record the structural rewrite, retained evidence boundary, and remaining manuscript limitations in the revision log.
- [x] **Step 3:** Run the focused test, then the full suite and the artifact verifier.

### Task 4: Review and hand off

**Files:**
- No new source files.

- [x] **Step 1:** Run `git diff --check` and inspect the scoped diff for accidental internal identifiers, unsupported claims, stale values, and broken image paths.
- [ ] **Step 2:** Commit the manuscript restructure separately from the earlier analysis commits, then push `codex/submission-audit`.
- [ ] **Step 3:** Report the new manuscript path, verification results, and the fact that the old draft remains available as an audit底稿.
