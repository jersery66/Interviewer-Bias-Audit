import pandas as pd

from reanalysis_v2.c5_blind_workflow import (
    DOMAIN_FEATURES,
    build_blind_cases,
    build_consensus_template,
    build_disagreement_sheet,
    build_line_index,
    build_review_form,
    compute_dp_vs_consensus,
    compute_interrater_metrics,
    make_blind_id_key,
    validate_consensus_labels,
    validate_reviewer_form,
)


def _cases():
    sample = pd.DataFrame(
        {
            "audit_case_id": ["A1", "A2"],
            "participant_id": [11, 12],
            "paper_label_phq8_ge10": [0, 1],
            "word_count": [2, 3],
            "length_stratum": ["short", "long"],
        }
    )
    participants = pd.DataFrame(
        {
            "participant_id": [11, 12],
            "text": ["sad\nno sleep", "I feel fine"],
        }
    )
    key = make_blind_id_key(sample)
    return build_blind_cases(participants, key), key


def _completed(reviewer_id: str, blind_ids):
    form = build_review_form(blind_ids, reviewer_id=reviewer_id, order_seed=1 if reviewer_id == "A" else 2)
    form["domain_label"] = "0"
    form["count_bin"] = "0"
    form["polarity"] = "ambiguous"
    form.loc[(form["blind_id"] == "B001") & (form["domain"] == DOMAIN_FEATURES[0]), "domain_label"] = "1"
    form.loc[(form["blind_id"] == "B001") & (form["domain"] == DOMAIN_FEATURES[0]), "count_bin"] = "1"
    form.loc[(form["blind_id"] == "B001") & (form["domain"] == DOMAIN_FEATURES[0]), "evidence_quote"] = "sad"
    form.loc[(form["blind_id"] == "B001") & (form["domain"] == DOMAIN_FEATURES[0]), "line_number"] = "1"
    form.loc[(form["blind_id"] == "B001") & (form["domain"] == DOMAIN_FEATURES[0]), "polarity"] = "present"
    return form


def test_blind_packet_has_line_index_and_reviewer_specific_order():
    cases, _ = _cases()
    lines = build_line_index(cases)
    assert list(cases.columns) == ["blind_id", "participant_text"]
    assert lines.loc[lines["blind_id"] == "B001", "line_number"].tolist() == [1, 2]
    a = build_review_form(cases["blind_id"], reviewer_id="A", order_seed=0)
    b = build_review_form(cases["blind_id"], reviewer_id="B", order_seed=10)
    assert list(a.columns) == [
        "reviewer_id", "blind_id", "domain", "domain_label", "count_bin",
        "evidence_quote", "line_number", "polarity", "uncertainty_reason",
    ]
    assert a["blind_id"].tolist() != b["blind_id"].tolist()
    validate_reviewer_form(a, expected_blind_ids=cases["blind_id"], require_completed=False)


def test_interrater_and_consensus_preserve_raw_values():
    cases, _ = _cases()
    a = _completed("A", cases["blind_id"])
    b = _completed("B", cases["blind_id"])
    cell = (b["blind_id"] == "B001") & (b["domain"] == DOMAIN_FEATURES[0])
    b.loc[cell, "domain_label"] = "NS"
    b.loc[cell, "count_bin"] = pd.NA
    b.loc[cell, "uncertainty_reason"] = "context unclear"
    validate_reviewer_form(a, expected_blind_ids=cases["blind_id"], require_completed=True)
    validate_reviewer_form(b, expected_blind_ids=cases["blind_id"], require_completed=True)
    metrics = compute_interrater_metrics(a, b)
    assert metrics.loc[metrics["scope"] == "overall", "three_category_n"].iat[0] == 20
    disagreements = build_disagreement_sheet(a, b, cases)
    assert len(disagreements) == 1
    template = build_consensus_template(a, b)
    template.loc[
        (template["blind_id"] == "B001") & (template["domain"] == DOMAIN_FEATURES[0]),
        ["consensus_label", "adjudication_reason"],
    ] = ["1", "quote supports symptom; NS not retained"]
    template.loc[~((template["blind_id"] == "B001") & (template["domain"] == DOMAIN_FEATURES[0])), "consensus_label"] = "0"
    template.loc[~((template["blind_id"] == "B001") & (template["domain"] == DOMAIN_FEATURES[0])), "adjudication_reason"] = "no supporting evidence"
    validate_consensus_labels(template, a, b)


def test_dp_vs_consensus_reports_ns_policy_and_domain_metrics():
    cases, _ = _cases()
    a = _completed("A", cases["blind_id"])
    b = _completed("B", cases["blind_id"])
    template = build_consensus_template(a, b)
    template["consensus_label"] = "0"
    template["adjudication_reason"] = "no evidence"
    target = (template["blind_id"] == "B001") & (template["domain"] == DOMAIN_FEATURES[0])
    template.loc[target, "consensus_label"] = "1"
    scoring = pd.DataFrame({"blind_id": ["B001", "B002"]})
    for domain in DOMAIN_FEATURES:
        scoring[domain] = 0
    scoring.loc[scoring["blind_id"] == "B001", DOMAIN_FEATURES[0]] = 2
    scoring.loc[scoring["blind_id"] == "B002", DOMAIN_FEATURES[1]] = 1
    result = compute_dp_vs_consensus(template, scoring, ns_policy="exclude")
    assert set(result["scope"]) == {"overall", "macro", *DOMAIN_FEATURES}
    assert result.loc[result["scope"] == "anhedonia_interest", "recall"].iat[0] == 1.0
