import numpy as np
import pandas as pd

from reanalysis_v2.controls import (
    DOMAIN_ORDER,
    build_domain_features,
    exclude_exact_quote_tokens,
    fold_safe_shuffle,
    keyword_mask,
    length_matched_sample,
    position_matched_turns,
)
from reanalysis_v2.embeddings import embed_documents, mean_pool_document, word_chunks


def test_c5_sampling_is_exact_reproducible_and_quote_excluding():
    text = "zero one two three four five six seven eight nine"
    first = length_matched_sample(text, target_words=4, seed=101)
    second = length_matched_sample(text, target_words=4, seed=101)
    assert first == second
    assert len(first.split()) == 4

    remaining, audit = exclude_exact_quote_tokens(text, ["two three", "seven eight"])
    assert remaining == "zero one four five six nine"
    assert audit["matched_quotes"] == 2
    assert audit["excluded_word_count"] == 4

    overlapping, overlap_audit = exclude_exact_quote_tokens(
        "a b c d e", ["b c", "c d"]
    )
    assert overlapping == "a e"
    assert overlap_audit["matched_quotes"] == 2
    assert overlap_audit["excluded_word_count"] == 3


def test_domain_features_and_keyword_mask_are_fixed():
    spans = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "domain": ["depressed_mood", "depressed_mood", "sleep_fatigue_energy"],
        }
    )
    presence, counts = build_domain_features(spans, participant_ids=[1, 2, 3])
    assert list(presence.columns) == ["participant_id", *DOMAIN_ORDER]
    assert presence.loc[presence.participant_id == 1, "depressed_mood"].item() == 1
    assert counts.loc[counts.participant_id == 1, "depressed_mood"].item() == 2
    assert presence.loc[presence.participant_id == 3, DOMAIN_ORDER].sum(axis=1).item() == 0

    masked, count = keyword_mask(
        "Depressed but not depression; tiredness and tired.", ["depressed", "tired"]
    )
    assert masked == "[MASKED] but not depression; tiredness and [MASKED]."
    assert count == 2


def test_c4_position_matching_and_fold_shuffle_do_not_cross_outer_roles():
    turns = pd.DataFrame(
        {
            "turn_index": [1, 2, 3, 4, 5],
            "normalized_spoken_text": ["aa bb", "cc", "dd ee ff", "gg", "hh ii"],
            "is_protocol_template_spoken": [0, 1, 0, 1, 0],
        }
    )
    matched = position_matched_turns(turns, target_words=2)
    assert len(matched.split()) == 2
    assert set(matched.split()).issubset({"aa", "bb", "dd", "ee", "ff", "hh", "ii"})

    ids = np.array([10, 11, 12, 13, 14, 15])
    texts = {participant_id: f"text-{participant_id}" for participant_id in ids}
    shuffled, mapping = fold_safe_shuffle(
        ids,
        texts,
        train_ids=np.array([10, 11, 12, 13]),
        test_ids=np.array([14, 15]),
        seed=44,
    )
    train = {10, 11, 12, 13}
    test = {14, 15}
    assert all((r.recipient_id in train) == (r.donor_id in train) for r in mapping.itertuples())
    assert all((r.recipient_id in test) == (r.donor_id in test) for r in mapping.itertuples())
    assert set(shuffled) == set(ids)


def test_embedding_chunking_covers_long_text_and_mean_pool_is_unweighted():
    chunks = word_chunks("a b c d e f g", chunk_words=4, overlap=2)
    assert chunks == ["a b c d", "c d e f", "e f g"]

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[len(text.split()), 1.0] for text in texts], dtype=np.float32)

    pooled = mean_pool_document("a b c d e f g", Encoder(), dim=2, chunk_words=4, overlap=2)
    expected = np.mean(np.array([[4.0, 1.0], [4.0, 1.0], [3.0, 1.0]]), axis=0)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(pooled, expected, rtol=1e-6)
    np.testing.assert_array_equal(mean_pool_document("", Encoder(), dim=2), np.zeros(2))


def test_batch_document_embedding_preserves_document_order_and_empty_rows():
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[len(text.split()), 1.0] for text in texts], dtype=np.float32)

    matrix, audit = embed_documents(
        ["a b c d e", "", "x y"],
        Encoder(),
        dim=2,
        chunk_words=3,
        overlap=1,
        batch_size=8,
    )
    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[1], np.zeros(2))
    assert audit.chunk_count.tolist() == [2, 0, 1]
    assert audit.word_count.tolist() == [5, 0, 2]
    assert audit.document_index.tolist() == [0, 1, 2]
