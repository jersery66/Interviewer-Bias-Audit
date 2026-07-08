#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate human coding → Table 2 + inter-rater agreement
========================================================
Module 13 downstream step. RUN THIS ONLY AFTER two human reviewers have
filled `coding_sheet_selected.csv` (columns r1_* and r2_*).

What it does
------------
1. Reads the filled coding sheet.
2. Produces Table 2: mismatch_type × primary attribution (R1-R6) counts,
   with the typical free-text descriptions and a fixed interpretation string.
3. Computes inter-rater agreement: agreement rate and Cohen's Kappa on
   r1_primary vs r2_primary (sklearn.metrics.cohen_kappa_score).
4. Writes `attribution_table2_by_attribution.csv` and prints a short report.

It does NOT invent any attribution; it only aggregates what reviewers entered.

Usage
-----
    python aggregate_coding.py            # reads coding_sheet_selected.csv in CWD
    python aggregate_coding.py path/to/filled.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Fixed interpretation strings per attribution code (do not edit casually).
INTERPRET = {
    "R1": "访谈覆盖不足：模型缺少支持阳性的文本线索，易低估",
    "R2": "表达不足或防御：回答短、症状未展开，模型易低估",
    "R3": "时间窗不一致：文本是否对应 PHQ-8 过去两周不清，方向不定",
    "R4": "负性事件非症状化：压力/创伤叙述多但非 PHQ-8 症状，模型易高估",
    "R5": "证据边界偏宽：泛化情绪词或非特异困扰被抽取，模型易高估",
    "R6": "自评—访谈通道不一致：量表与访谈表达明显不符，方向不定",
}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "coding_sheet_selected.csv"
    df = pd.read_csv(src)

    need = ["r1_primary", "r2_primary", "mismatch_type"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少列：{missing}（请确认编码表已填写）")

    filled = df[df["r1_primary"].notna() & (df["r1_primary"].astype(str).str.strip() != "")]
    if len(filled) == 0:
        raise SystemExit("未检测到已填写的 r1_primary，请先由复核者填写编码表。")

    # ── Table 2 ─────────────────────────────────────────────────────────────
    rows = []
    for (mt, code), grp in filled.groupby(["mismatch_type", "r1_primary"]):
        texts = grp["r1_text"].dropna().astype(str)
        texts = texts[texts.str.strip() != ""]
        typical = "；".join(texts.head(3).tolist())
        rows.append({
            "mismatch_type": mt,
            "primary_attribution": code,
            "n": len(grp),
            "typical_text": typical,
            "interpretation": INTERPRET.get(str(code).strip(), ""),
        })
    table2 = pd.DataFrame(rows).sort_values(["mismatch_type", "n"],
                                          ascending=[True, False]).reset_index(drop=True)
    table2.to_csv("attribution_table2_by_attribution.csv", index=False, encoding="utf-8-sig")
    print("[TABLE2] written attribution_table2_by_attribution.csv")
    print(table2.to_string(index=False))

    # ── Inter-rater agreement ────────────────────────────────────────────────
    r1 = filled["r1_primary"].astype(str).str.strip()
    r2 = filled["r2_primary"].astype(str).str.strip()
    both = r1.notna() & r2.notna() & (r1 != "") & (r2 != "")
    if both.sum() >= 2:
        a = r1[both].values
        b = r2[both].values
        agree = float((a == b).mean())
        # Cohen's Kappa requires integer/finite labels; map codes to indices
        labels = pd.unique(np.concatenate([a, b]))
        map_ = {lab: i for i, lab in enumerate(labels)}
        kappa = cohen_kappa_score([map_[x] for x in a], [map_[x] for x in b])
        print(f"\n[AGREEMENT] reviewers both coded n={both.sum()}")
        print(f"             agreement rate = {agree:.3f}")
        print(f"             Cohen's Kappa = {kappa:.3f}")
    else:
        print("\n[AGREEMENT] 不足两例同时含 r1/r2，跳过 Kappa。")


if __name__ == "__main__":
    main()
