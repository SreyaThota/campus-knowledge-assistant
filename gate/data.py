"""Data loading, binary mapping, and split integrity utilities for the scope gate.

Binary task definition:
- Label 1: In-scope (any of the 150 standard intents in CLINC150)
- Label 0: Out-of-scope (OOS intent)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Canonical binary class labels
LABEL_IN_SCOPE: int = 1
LABEL_OOS: int = 0

CLINC_OOS_INTENT_NAME: str = "oos"
CLINC_OOS_INTENT_INDEX: int = 42

DEFAULT_CACHE_PATH = Path("data/cache/clinc_binary.json")


def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison and deduplication."""
    return text.strip().lower()


def load_clinc_data(
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
    dedup_leaks: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CLINC150 ('clinc_oos', 'plus') and convert to binary in-scope/out-of-scope task.

    Args:
        cache_path: Path to local JSON cache file.
        use_cache: If True, load from cache if available, or populate cache after loading.
        dedup_leaks: If True, remove the 5 raw leaked training examples that duplicate
                     validation or test queries in official CLINC150.

    Returns:
        Dict with keys 'train', 'validation', 'test', each containing a list of
        {'text': str, 'label': int, 'intent': str or int}.
    """
    path = cache_path or DEFAULT_CACHE_PATH

    if use_cache and path.exists():
        logger.info(f"Loading CLINC150 binary dataset from local cache: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    import datasets

    logger.info("Loading CLINC150 ('plus') from datasets library...")
    raw_ds = datasets.load_dataset("clinc/clinc_oos", "plus")

    splits: Dict[str, List[Dict[str, Any]]] = {}

    for split_name in ["train", "validation", "test"]:
        split_items = []
        for row in raw_ds[split_name]:
            text = row["text"].strip()
            raw_intent = row["intent"]

            # Intent 42 in CLINC 'plus' corresponds to 'oos'
            is_oos = (raw_intent == CLINC_OOS_INTENT_INDEX) or (raw_intent == CLINC_OOS_INTENT_NAME)
            binary_label = LABEL_OOS if is_oos else LABEL_IN_SCOPE

            split_items.append({
                "text": text,
                "label": binary_label,
                "raw_intent": raw_intent,
            })
        splits[split_name] = split_items

    if dedup_leaks:
        val_texts = {normalize_text(x["text"]) for x in splits["validation"]}
        test_texts = {normalize_text(x["text"]) for x in splits["test"]}
        leak_set = val_texts.union(test_texts)

        original_train_count = len(splits["train"])
        splits["train"] = [
            x for x in splits["train"]
            if normalize_text(x["text"]) not in leak_set
        ]
        leaks_removed = original_train_count - len(splits["train"])
        if leaks_removed > 0:
            logger.info(
                f"Removed {leaks_removed} leaked queries from training split that overlapped "
                f"with validation or test sets."
            )

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Saved CLINC150 binary dataset to cache: {path}")

    return splits


def get_split_stats(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute total sample counts, class distributions, and OOS ratios per split.
    """
    stats = {}
    for name, items in splits.items():
        total = len(items)
        in_scope = sum(1 for x in items if x["label"] == LABEL_IN_SCOPE)
        oos = sum(1 for x in items if x["label"] == LABEL_OOS)
        stats[name] = {
            "total": total,
            "in_scope": in_scope,
            "oos": oos,
            "oos_ratio": round(oos / total, 4) if total > 0 else 0.0,
            "imbalance_ratio": round(in_scope / oos, 2) if oos > 0 else float("inf"),
        }
    return stats


def assert_splits_disjoint(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Assert that train, validation, and test splits have 0 query text overlap.
    """
    train_texts = {normalize_text(x["text"]) for x in splits["train"]}
    val_texts = {normalize_text(x["text"]) for x in splits["validation"]}
    test_texts = {normalize_text(x["text"]) for x in splits["test"]}

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    assert not train_val_overlap, f"Train and Validation overlap on: {train_val_overlap}"
    assert not train_test_overlap, f"Train and Test overlap on: {train_test_overlap}"
    assert not val_test_overlap, f"Validation and Test overlap on: {val_test_overlap}"


def create_synthetic_splits() -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates small synthetic disjoint splits for unit tests without network or disk requirements.
    """
    return {
        "train": [
            {"text": f"synthetic in-scope training query {i}", "label": LABEL_IN_SCOPE, "raw_intent": i % 10}
            for i in range(20)
        ] + [
            {"text": f"synthetic out-of-scope training query {i}", "label": LABEL_OOS, "raw_intent": CLINC_OOS_INTENT_INDEX}
            for i in range(2)
        ],
        "validation": [
            {"text": f"synthetic in-scope validation query {i}", "label": LABEL_IN_SCOPE, "raw_intent": i % 10}
            for i in range(10)
        ] + [
            {"text": f"synthetic out-of-scope validation query {i}", "label": LABEL_OOS, "raw_intent": CLINC_OOS_INTENT_INDEX}
            for i in range(2)
        ],
        "test": [
            {"text": f"synthetic in-scope test query {i}", "label": LABEL_IN_SCOPE, "raw_intent": i % 10}
            for i in range(10)
        ] + [
            {"text": f"synthetic out-of-scope test query {i}", "label": LABEL_OOS, "raw_intent": CLINC_OOS_INTENT_INDEX}
            for i in range(5)
        ],
    }
