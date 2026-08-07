# Figure contract — strict source-importance manuscript

## Claim hierarchy

- Primary supported claim: participant text alone has moderate repeated-OOF discrimination; in M3, domain-count group permutation importance is significant after BH-FDR (`q = 0.0039`, `**`).
- Required negative result: all seven incremental model comparisons are `ns` after BH-FDR.
- Exploratory result: all 20 domain leave-one-out comparisons are `ns`; domain rankings must not be described as confirmed importance.
- Prohibited claim: protocol presence and domain counts are jointly established as the main sources, or M3 significantly improves over M0 after multiplicity correction.

## Figure inventory

| Figure | Archetype | Purpose | Statistical annotation |
|---|---|---|---|
| Figure 1 | Method schematic | Show fold-local preprocessing, direct raw-source joint models, participant-level inference and multiplicity control | Only the domain-count importance result carries `**`; all model increments are stated as `ns` |
| Figure 2 | Two-panel forest plot | Report model AUCs and all seven paired incremental comparisons | 95% participant-bootstrap CI plus BH-FDR `q`; all comparisons `ns` |
| Figure 3 | Group permutation forest plot | Report conditional source importance in M3 | Domain counts `q = 0.0039`, `**`; template presence and participant text `ns` |
| Figure 4 | Two-panel domain LOO forest plot | Show count/presence domain ablations without confirmatory ranking | 20-test BH-FDR family; all `ns` |

## Output contract

- Backend: Python/matplotlib with non-interactive `Agg` backend.
- Width: double-column manuscript layout, maximum 7.2 inches (approximately 183 mm).
- Formats: editable-text SVG, TrueType-embedded PDF, LZW-compressed 600 dpi TIFF and 240 dpi PNG preview.
- Source tables: `analysis_v2/10_source_importance/07_strict_joint_models/`.
- Color: restrained blue/orange/teal/purple palette; significance is never encoded by color alone.
- Captions: each figure is embedded immediately before its caption in `manuscript_draft_zh.md`.
