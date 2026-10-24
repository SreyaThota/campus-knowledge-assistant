"""Script to compare 4 baselines and 3 learned-gate seeds strictly on the CLINC validation split.

Thresholds are tuned strictly on validation (criterion='max_f1').
Test split is NOT loaded or touched.
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.data import load_clinc_data, LABEL_IN_SCOPE, LABEL_OOS
from gate.baselines import (
    ExemplarCosineBaseline,
    CentroidCosineBaseline,
    TfidfLogisticRegressionBaseline,
    FrozenEmbeddingLogisticRegressionBaseline,
    compute_binary_metrics_at_threshold,
    find_optimal_threshold,
)
from gate.predict import ScopeGatePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("val_comparison")


def main():
    logger.info("Loading CLINC train and validation splits...")
    splits = load_clinc_data(use_cache=True, dedup_leaks=True)

    train_texts = [x["text"] for x in splits["train"]]
    train_labels = [x["label"] for x in splits["train"]]

    val_texts = [x["text"] for x in splits["validation"]]
    val_labels = np.array([x["label"] for x in splits["validation"]], dtype=int)

    logger.info(f"Loaded {len(train_texts)} train items, {len(val_texts)} validation items.")

    results = []

    # 1. TF-IDF + Logistic Regression
    logger.info("--- Fitting 1. TF-IDF + Logistic Regression ---")
    tfidf_model = TfidfLogisticRegressionBaseline(seed=42)
    tfidf_model.fit(train_texts, train_labels)
    tfidf_scores = tfidf_model.predict_oos_score(val_texts)
    th_tfidf = find_optimal_threshold(tfidf_scores, val_labels, criterion="max_f1")
    m_tfidf = compute_binary_metrics_at_threshold(tfidf_scores, val_labels, th_tfidf)
    results.append({
        "method": "TF-IDF + Logistic Regression",
        "threshold": th_tfidf,
        "oos_f1": m_tfidf["oos_f1"],
        "oos_recall": m_tfidf["oos_recall"],
        "oos_precision": m_tfidf["oos_precision"],
        "in_scope_acceptance": m_tfidf["in_scope_acceptance"],
        "tp": m_tfidf["tp"],
        "fp": m_tfidf["fp"],
        "fn": m_tfidf["fn"],
        "tn": m_tfidf["tn"],
    })

    # For sentence-transformers baselines, we can share the encoder to avoid reloading model
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Pre-encoding train texts for embedding baselines...")
    train_in_scope_texts = [t for t, l in zip(train_texts, train_labels) if l == LABEL_IN_SCOPE]
    train_in_scope_embs = encoder.encode(
        train_in_scope_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    logger.info("Pre-encoding validation texts...")
    val_embs = encoder.encode(
        val_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )

    # 2. Centroid Cosine Similarity
    logger.info("--- Evaluating 2. Centroid Cosine Baseline ---")
    centroid = np.mean(train_in_scope_embs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    centroid_scores = 1.0 - np.dot(val_embs, centroid)
    centroid_scores = np.clip(centroid_scores, 0.0, 1.0)
    th_centroid = find_optimal_threshold(centroid_scores, val_labels, criterion="max_f1")
    m_centroid = compute_binary_metrics_at_threshold(centroid_scores, val_labels, th_centroid)
    results.append({
        "method": "Centroid Cosine Similarity",
        "threshold": th_centroid,
        "oos_f1": m_centroid["oos_f1"],
        "oos_recall": m_centroid["oos_recall"],
        "oos_precision": m_centroid["oos_precision"],
        "in_scope_acceptance": m_centroid["in_scope_acceptance"],
        "tp": m_centroid["tp"],
        "fp": m_centroid["fp"],
        "fn": m_centroid["fn"],
        "tn": m_centroid["tn"],
    })

    # 3. Exemplar Cosine Similarity
    logger.info("--- Evaluating 3. Exemplar Cosine Baseline ---")
    # Batch dot product
    sim_matrix = np.dot(val_embs, train_in_scope_embs.T)
    max_sims = np.max(sim_matrix, axis=1)
    exemplar_scores = np.clip(1.0 - max_sims, 0.0, 1.0)
    th_exemplar = find_optimal_threshold(exemplar_scores, val_labels, criterion="max_f1")
    m_exemplar = compute_binary_metrics_at_threshold(exemplar_scores, val_labels, th_exemplar)
    results.append({
        "method": "Exemplar Cosine Similarity (Nearest Neighbor)",
        "threshold": th_exemplar,
        "oos_f1": m_exemplar["oos_f1"],
        "oos_recall": m_exemplar["oos_recall"],
        "oos_precision": m_exemplar["oos_precision"],
        "in_scope_acceptance": m_exemplar["in_scope_acceptance"],
        "tp": m_exemplar["tp"],
        "fp": m_exemplar["fp"],
        "fn": m_exemplar["fn"],
        "tn": m_exemplar["tn"],
    })

    # 4. Frozen Embedding + Logistic Regression
    logger.info("--- Fitting 4. Frozen Embedding + Logistic Regression ---")
    logger.info("Encoding all training texts...")
    train_all_embs = encoder.encode(
        train_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(train_all_embs, np.array(train_labels, dtype=int))
    proba = clf.predict_proba(val_embs)
    oos_col = list(clf.classes_).index(LABEL_OOS)
    frozen_logreg_scores = proba[:, oos_col]
    th_frozen = find_optimal_threshold(frozen_logreg_scores, val_labels, criterion="max_f1")
    m_frozen = compute_binary_metrics_at_threshold(frozen_logreg_scores, val_labels, th_frozen)
    results.append({
        "method": "Frozen Embedding + Logistic Regression",
        "threshold": th_frozen,
        "oos_f1": m_frozen["oos_f1"],
        "oos_recall": m_frozen["oos_recall"],
        "oos_precision": m_frozen["oos_precision"],
        "in_scope_acceptance": m_frozen["in_scope_acceptance"],
        "tp": m_frozen["tp"],
        "fp": m_frozen["fp"],
        "fn": m_frozen["fn"],
        "tn": m_frozen["tn"],
    })

    # 5. Learned Scope Gate Seeds
    seeds = [42, 1337, 2026]
    learned_metrics = []

    for s in seeds:
        ckpt_path = Path(f"results/checkpoints/scope_gate_seed_{s}.pt")
        logger.info(f"--- Evaluating Learned Scope Gate Seed {s} ({ckpt_path}) ---")
        pred = ScopeGatePredictor(ckpt_path)
        scores = pred.predict_oos_score(val_texts)
        # Using tuned threshold recorded in checkpoint metadata or re-tuning
        th = pred.threshold
        m = compute_binary_metrics_at_threshold(scores, val_labels, th)
        learned_metrics.append(m)
        results.append({
            "method": f"Learned Gate (Seed {s})",
            "threshold": th,
            "oos_f1": m["oos_f1"],
            "oos_recall": m["oos_recall"],
            "oos_precision": m["oos_precision"],
            "in_scope_acceptance": m["in_scope_acceptance"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Method':<44} | {'Threshold':<9} | {'OOS F1':<8} | {'OOS Recall':<10} | {'OOS Prec':<8} | {'In-Scope Acc':<12}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['method']:<44} | {r['threshold']:<9.4f} | {r['oos_f1']:<8.4f} | "
            f"{r['oos_recall']:<10.4f} | {r['oos_precision']:<8.4f} | {r['in_scope_acceptance']:<12.4f}"
        )
    print("-" * 105)

    f1s = [m["oos_f1"] for m in learned_metrics]
    recalls = [m["oos_recall"] for m in learned_metrics]
    precs = [m["oos_precision"] for m in learned_metrics]
    in_accs = [m["in_scope_acceptance"] for m in learned_metrics]

    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
    rec_mean, rec_std = np.mean(recalls), np.std(recalls)
    prec_mean, prec_std = np.mean(precs), np.std(precs)
    acc_mean, acc_std = np.mean(in_accs), np.std(in_accs)

    mean_str = f"Learned Gate (Mean \u00b1 Std across 3 seeds)"
    f1_str = f"{f1_mean:.4f} \u00b1 {f1_std:.4f}"
    rec_str = f"{rec_mean:.4f} \u00b1 {rec_std:.4f}"
    prec_str = f"{prec_mean:.4f} \u00b1 {prec_std:.4f}"
    acc_str = f"{acc_mean:.4f} \u00b1 {acc_std:.4f}"

    print(f"{mean_str:<44} | {'--':<9} | {f1_str:<8} | {rec_str:<10} | {prec_str:<8} | {acc_str:<12}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
