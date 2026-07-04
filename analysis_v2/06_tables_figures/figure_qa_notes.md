# Figure QA notes

- Backend exclusivity: PASS; all exports were generated with Python/matplotlib.
- Final dimensions: Figure 1, 183 x 67 mm; Figure 2, 183 x 126 mm; Figure S1, 183 x 73 mm; Figure S2, 125 x 74 mm before tight bounding-box adjustment.
- Export bundle: PASS; SVG, PDF, 600-dpi TIFF, 240-dpi PNG, source tables and legends are present.
- Editable text: SVG uses `svg.fonttype=none`; PDF uses TrueType font embedding (`pdf.fonttype=42`).
- Font: Arial/Helvetica requested with DejaVu Sans fallback.
- Color: representation is encoded by color and marker/line identity; no rainbow map; red/green is not used as the sole encoding.
- Statistics: n, split design, aggregation level, permutation count and FDR family are stated in legends/reports.
- Source data: each quantitative figure maps to CSV tables in this package or the frozen computational output referenced by the run manifest.
- Image integrity: not applicable; no microscopy, blots or photographic panels.
- Remaining submission check: verify exact journal-specific size/file rules at submission time because author guidance can change.
