import numpy as np

from reanalysis_v2.modeling import (
    fit_predict_dense,
    fit_predict_tfidf,
    make_inner_splits,
    tune_dense_c,
)


def test_tfidf_vocabulary_is_fit_on_outer_training_only():
    train_texts = [
        "stable alpha alpha",
        "stable alpha",
        "stable beta beta",
        "stable beta",
        "stable gamma",
        "stable gamma gamma",
    ]
    y_train = np.array([0, 0, 1, 1, 0, 1])
    test_texts = ["testonlytoken alpha", "testonlytoken beta"]
    probability, fitted = fit_predict_tfidf(train_texts, y_train, test_texts, seed=3)
    assert probability.shape == (2,)
    assert "testonlytoken" not in fitted.named_steps["tfidf"].vocabulary_


def test_inner_splits_are_relative_to_outer_training_and_deterministic():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    first = make_inner_splits(y, n_splits=3, seed=9)
    second = make_inner_splits(y, n_splits=3, seed=9)
    assert len(first) == 3
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        assert set(train_a).isdisjoint(test_a)
        assert set(train_a) | set(test_a) == set(range(len(y)))


def test_dense_c_tuning_and_scaling_never_use_outer_test_data():
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(30, 5))
    y_train = np.array([0, 1] * 15)
    X_test = np.full((3, 5), 10_000.0)
    inner = make_inner_splits(y_train, n_splits=3, seed=11)
    selected_c, scores = tune_dense_c(
        X_train, y_train, inner, c_grid=[0.01, 0.1, 1.0], seed=12
    )
    assert selected_c in {0.01, 0.1, 1.0}
    assert set(scores) == {0.01, 0.1, 1.0}

    probability, fitted = fit_predict_dense(
        X_train, y_train, X_test, c=selected_c, seed=12
    )
    assert probability.shape == (3,)
    np.testing.assert_allclose(
        fitted.named_steps["scale"].mean_, X_train.mean(axis=0), rtol=1e-12, atol=1e-12
    )
