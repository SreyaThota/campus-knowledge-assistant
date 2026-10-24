"""Unit tests for gate.data split integrity and binary mapping."""

import pytest
from gate.data import (
    load_clinc_data,
    get_split_stats,
    assert_splits_disjoint,
    create_synthetic_splits,
    LABEL_IN_SCOPE,
    LABEL_OOS,
)


def test_synthetic_splits_integrity():
    """Verify synthetic split generator produces disjoint binary splits."""
    splits = create_synthetic_splits()
    assert_splits_disjoint(splits)

    for split_name, items in splits.items():
        assert len(items) > 0
        for item in items:
            assert item["label"] in [LABEL_IN_SCOPE, LABEL_OOS]
            assert isinstance(item["text"], str)


def test_split_disjointness_catches_leak():
    """Ensure assert_splits_disjoint raises AssertionError when overlap exists."""
    splits = create_synthetic_splits()
    # Artificially inject a leak from test into train
    leaked_text = splits["test"][0]["text"]
    splits["train"].append({"text": leaked_text, "label": LABEL_IN_SCOPE})

    with pytest.raises(AssertionError) as exc_info:
        assert_splits_disjoint(splits)
    assert "overlap" in str(exc_info.value).lower()


def test_cached_clinc_splits_integrity():
    """Verify cached CLINC150 binary splits have zero cross-split leakage."""
    splits = load_clinc_data(use_cache=True)
    assert_splits_disjoint(splits)

    stats = get_split_stats(splits)
    # Train set: ~15k in-scope, 250 OOS
    assert stats["train"]["oos"] == 250
    assert stats["train"]["in_scope"] == 14995  # 15000 minus 5 leaked duplicates
    assert stats["train"]["total"] == 15245

    # Validation set: 3000 in-scope, 100 OOS
    assert stats["validation"]["oos"] == 100
    assert stats["validation"]["in_scope"] == 3000
    assert stats["validation"]["total"] == 3100

    # Test set: 4500 in-scope, 1000 OOS
    assert stats["test"]["oos"] == 1000
    assert stats["test"]["in_scope"] == 4500
    assert stats["test"]["total"] == 5500


def test_binary_labels_validity():
    """Verify every sample across all splits has label strictly in {0, 1}."""
    splits = load_clinc_data(use_cache=True)
    valid_labels = {LABEL_IN_SCOPE, LABEL_OOS}
    for split_name, items in splits.items():
        labels = {item["label"] for item in items}
        assert labels == valid_labels, f"Split {split_name} has invalid labels: {labels}"
