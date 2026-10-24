"""Offline unit tests for gate.evaluate module.

Strictly zero-network execution: tests metrics, calibration, bootstrap CIs,
campus hash verification, and synthetic datasets offline.
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest

from gate.data import LABEL_IN_SCOPE, LABEL_OOS
from gate.evaluate import (
    compute_ece,
    plot_reliability_diagram,
    compute_bootstrap_confidence_intervals,
    compute_file_sha256,
    verify_campus_hash,
    load_campus_dataset,
    evaluate_single_method,
    compute_aggregate_seed_stats,
)


def test_compute_ece_perfect_and_imperfect():
    """Verify ECE calculation on known synthetic probabilities."""
    # 10 samples: 5 true in-scope (1), 5 true OOS (0)
    labels = np.array([LABEL_IN_SCOPE] * 5 + [LABEL_OOS] * 5)
    # Perfectly separated confidence
    probs_perfect = np.array([0.05] * 5 + [0.95] * 5)

    ece_perf, details_perf = compute_ece(probs_perfect, labels, num_bins=10)
    assert 0.0 <= ece_perf <= 0.1
    assert len(details_perf["bin_confs"]) == 10
    assert len(details_perf["bin_accs"]) == 10
    assert sum(details_perf["bin_counts"]) == 10

    # Totally uncalibrated confidence (inverted)
    probs_inverted = np.array([0.95] * 5 + [0.05] * 5)
    ece_inv, _ = compute_ece(probs_inverted, labels, num_bins=10)
    assert ece_inv >= 0.8


def test_reliability_diagram_generation(tmp_path: Path):
    """Verify reliability diagram plot generation runs offline and outputs PNG file."""
    labels = np.array([LABEL_IN_SCOPE] * 20 + [LABEL_OOS] * 10)
    scores = np.concatenate([
        np.random.RandomState(42).uniform(0.0, 0.4, 20),
        np.random.RandomState(42).uniform(0.6, 1.0, 10),
    ])
    ece, details = compute_ece(scores, labels, num_bins=5)

    out_file = tmp_path / "test_diagram.png"
    plot_reliability_diagram(details, ece, model_name="Test Model", output_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 1000  # Valid non-empty PNG


def test_bootstrap_confidence_intervals_bounds():
    """Verify 95% bootstrap CI yields valid [lower, upper] intervals."""
    rng = np.random.RandomState(42)
    labels = np.array([LABEL_IN_SCOPE] * 80 + [LABEL_OOS] * 20)
    scores = np.concatenate([
        rng.uniform(0.05, 0.35, 80),
        rng.uniform(0.55, 0.95, 20),
    ])

    ci_dict = compute_bootstrap_confidence_intervals(
        scores, labels, threshold=0.5, n_bootstraps=1000, ci=0.95, seed=42
    )

    for metric_name in ["oos_f1", "oos_recall", "oos_precision", "in_scope_acceptance"]:
        assert metric_name in ci_dict
        low, high = ci_dict[metric_name]
        assert 0.0 <= low <= high <= 1.0


def test_campus_hash_verification_refuses_when_missing_or_mismatch(tmp_path: Path):
    """Verify campus dataset security check strictly enforces frozen hash."""
    dummy_csv = tmp_path / "dummy_campus.csv"
    dummy_csv.write_text("question,label,source_doc,lang,difficulty,author,notes\n", encoding="utf-8")
    dummy_hash_file = tmp_path / "dummy_hash.sha256"

    # Case 1: Hash file does not exist -> must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Campus hash file not found"):
        verify_campus_hash(dummy_csv, dummy_hash_file)

    # Case 2: Hash file contains mismatching hash -> must raise ValueError
    dummy_hash_file.write_text("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_campus_hash(dummy_csv, dummy_hash_file)

    # Case 3: Matching hash -> must return valid hash
    correct_hash = compute_file_sha256(dummy_csv)
    dummy_hash_file.write_text(correct_hash, encoding="utf-8")
    returned_hash = verify_campus_hash(dummy_csv, dummy_hash_file)
    assert returned_hash == correct_hash


def test_load_campus_dataset_schema_and_author_filtering(tmp_path: Path):
    """Verify campus dataset parser handles authors and labels offline."""
    csv_content = (
        "question,label,source_doc,lang,difficulty,author,notes\n"
        "What are library hours?,in,doc1.txt,en,easy,ai_draft,Note 1\n"
        "How to bake bread?,out,None,en,easy,ai_draft,Note 2\n"
        "How do I submit exam waiver?,in,doc2.txt,en,hard,human,Note 3\n"
        "Who won World Cup?,out,None,en,easy,human,Note 4\n"
    )
    campus_csv = tmp_path / "campus_questions.csv"
    campus_csv.write_text(csv_content, encoding="utf-8")
    hash_file = tmp_path / "campus.sha256"
    hash_file.write_text(compute_file_sha256(campus_csv), encoding="utf-8")

    rows = load_campus_dataset(campus_csv, hash_file, verify_hash=True)
    assert len(rows) == 4

    human_rows = [r for r in rows if r["author"] == "human"]
    ai_rows = [r for r in rows if r["author"] == "ai_draft"]
    assert len(human_rows) == 2
    assert len(ai_rows) == 2

    assert rows[0]["label"] == LABEL_IN_SCOPE
    assert rows[1]["label"] == LABEL_OOS
    assert rows[2]["label"] == LABEL_IN_SCOPE
    assert rows[3]["label"] == LABEL_OOS


def test_evaluate_single_method_end_to_end_synthetic(tmp_path: Path):
    """Verify end-to-end evaluation of single method on synthetic predictions."""
    labels = np.array([LABEL_IN_SCOPE] * 50 + [LABEL_OOS] * 50)
    scores = np.concatenate([
        np.random.RandomState(42).uniform(0.0, 0.4, 50),
        np.random.RandomState(42).uniform(0.6, 1.0, 50),
    ])

    res = evaluate_single_method(
        name="Synthetic Model",
        scores=scores,
        labels=labels,
        threshold=0.5,
        n_bootstraps=1000,
        seed=42,
        save_reliability_diagram_path=tmp_path / "diag.png",
    )

    assert res["method"] == "Synthetic Model"
    assert res["metrics"]["oos_f1"] == 1.0
    assert res["metrics"]["in_scope_acceptance"] == 1.0
    assert "confidence_intervals_95" in res
    assert "ece" in res["metrics"]


def test_compute_aggregate_seed_stats():
    """Verify mean +- std computation across seed evaluations."""
    evals = [
        {"threshold": 0.30, "metrics": {"oos_f1": 0.70, "oos_recall": 0.60, "oos_precision": 0.80, "in_scope_acceptance": 0.99, "ece": 0.05}},
        {"threshold": 0.35, "metrics": {"oos_f1": 0.74, "oos_recall": 0.65, "oos_precision": 0.85, "in_scope_acceptance": 0.995, "ece": 0.04}},
        {"threshold": 0.40, "metrics": {"oos_f1": 0.72, "oos_recall": 0.61, "oos_precision": 0.81, "in_scope_acceptance": 0.992, "ece": 0.06}},
    ]

    stats = compute_aggregate_seed_stats(evals)
    assert np.isclose(stats["oos_f1_mean"], 0.72, atol=1e-3)
    assert stats["oos_f1_std"] > 0
    assert np.isclose(stats["threshold_mean"], 0.35, atol=1e-3)
