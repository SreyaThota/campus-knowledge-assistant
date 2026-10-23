"""Unit tests for gate.baselines models and threshold tuning logic."""

import numpy as np
import pytest
from gate.data import LABEL_IN_SCOPE, LABEL_OOS
from gate.baselines import (
    compute_binary_metrics_at_threshold,
    find_optimal_threshold,
    TfidfLogisticRegressionBaseline,
)


def test_compute_binary_metrics_at_threshold():
    """Verify precision, recall, F1, and in-scope retention math."""
    # 2 true in-scope (1), 2 true OOS (0)
    labels = np.array([LABEL_IN_SCOPE, LABEL_IN_SCOPE, LABEL_OOS, LABEL_OOS])
    # OOS scores: in-scope low, OOS high
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    m = compute_binary_metrics_at_threshold(scores, labels, threshold=0.5)
    assert m["tp"] == 2  # both OOS correctly rejected
    assert m["fp"] == 0  # no in-scope rejected
    assert m["fn"] == 0  # no OOS accepted
    assert m["tn"] == 2  # both in-scope accepted
    assert m["oos_recall"] == 1.0
    assert m["oos_precision"] == 1.0
    assert m["oos_f1"] == 1.0
    assert m["in_scope_acceptance"] == 1.0


def test_find_optimal_threshold_synthetic():
    """Verify threshold tuner identifies optimal boundary."""
    labels = np.array([LABEL_IN_SCOPE] * 20 + [LABEL_OOS] * 10)
    scores = np.concatenate([
        np.random.RandomState(42).uniform(0.0, 0.4, 20),
        np.random.RandomState(42).uniform(0.6, 1.0, 10),
    ])

    best_th = find_optimal_threshold(scores, labels, criterion="max_f1")
    assert 0.35 <= best_th <= 0.65
    m = compute_binary_metrics_at_threshold(scores, labels, threshold=best_th)
    assert m["oos_f1"] == 1.0


def test_tfidf_baseline_end_to_end_synthetic():
    """Verify TF-IDF + Logistic Regression train, val threshold, and prediction."""
    train_texts = [
        "enroll in courses", "library study rooms", "register for exam", "grade transcripts",
        "make pizza dough", "weather in tokyo", "quantum physics theorem", "buy groceries"
    ]
    train_labels = [LABEL_IN_SCOPE] * 4 + [LABEL_OOS] * 4

    val_texts = ["exam schedule", "bake banana bread"]
    val_labels = [LABEL_IN_SCOPE, LABEL_OOS]

    baseline = TfidfLogisticRegressionBaseline(seed=42)
    baseline.fit(train_texts, train_labels)

    th = baseline.tune_threshold_on_validation(val_texts, val_labels, criterion="max_f1")
    assert 0.0 <= th <= 1.0
    assert baseline.val_threshold == th

    preds = baseline.predict(["study room reservation", "cook delicious pasta"])
    assert preds[0] == LABEL_IN_SCOPE
    assert preds[1] == LABEL_OOS
