"""Script to compute:
1. Validation threshold sweep (tau = 0.20 to 0.50, step 0.05) across 3 seeds.
2. 95% bootstrap confidence intervals on validation split for Exemplar Cosine and Learned Gate seeds.
3. False rejection and false acceptance counts across all methods.

Strictly validation split only (no test or campus data accessed).
"""

import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.data import load_clinc_data, LABEL_IN_SCOPE, LABEL_OOS
from gate.baselines import (
    ExemplarCosineBaseline,
    compute_binary_metrics_at_threshold,
)
from gate.predict import ScopeGatePredictor
from gate.evaluate import compute_bootstrap_confidence_intervals


def main():
    print("Loading validation split...")
    splits = load_clinc_data(use_cache=True, dedup_leaks=True)
    val_texts = [x["text"] for x in splits["validation"]]
    val_labels = np.array([x["label"] for x in splits["validation"]], dtype=int)
    print(f"Validation items: {len(val_texts)} ({np.sum(val_labels == LABEL_IN_SCOPE)} in-scope, {np.sum(val_labels == LABEL_OOS)} OOS)")

    # 1. Load 3 learned gate predictors
    seeds = [42, 1337, 2026]
    predictors = {}
    scores_dict = {}

    for s in seeds:
        ckpt_path = Path(f"results/checkpoints/scope_gate_seed_{s}.pt")
        pred = ScopeGatePredictor(ckpt_path)
        predictors[s] = pred
        scores = pred.predict_oos_score(val_texts)
        scores_dict[s] = scores

    # Threshold sweep from 0.20 to 0.50 in steps of 0.05
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_results = []

    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP ON VALIDATION SPLIT (tau from 0.20 to 0.50)")
    print("=" * 90)

    for th in thresholds:
        seed_metrics = []
        for s in seeds:
            m = compute_binary_metrics_at_threshold(scores_dict[s], val_labels, threshold=th)
            seed_metrics.append(m)

        f1s = [m["oos_f1"] for m in seed_metrics]
        recalls = [m["oos_recall"] for m in seed_metrics]
        precs = [m["oos_precision"] for m in seed_metrics]
        in_accs = [m["in_scope_acceptance"] for m in seed_metrics]
        fps = [m["fp"] for m in seed_metrics]
        fns = [m["fn"] for m in seed_metrics]

        sweep_results.append({
            "threshold": th,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "prec_mean": float(np.mean(precs)),
            "prec_std": float(np.std(precs)),
            "in_acc_mean": float(np.mean(in_accs)),
            "in_acc_std": float(np.std(in_accs)),
            "fp_mean": float(np.mean(fps)),
            "fp_std": float(np.std(fps)),
            "fn_mean": float(np.mean(fns)),
            "fn_std": float(np.std(fns)),
            "per_seed": {
                s: m for s, m in zip(seeds, seed_metrics)
            }
        })

    print(f"{'Threshold (tau)':<16} | {'OOS Recall':<18} | {'OOS Precision':<18} | {'OOS F1':<18} | {'In-Scope Acc':<18} | {'False Rejects (FP)':<18}")
    print("-" * 115)
    for row in sweep_results:
        rec_str = f"{row['recall_mean']:.4f} \u00b1 {row['recall_std']:.4f}"
        prec_str = f"{row['prec_mean']:.4f} \u00b1 {row['prec_std']:.4f}"
        f1_str = f"{row['f1_mean']:.4f} \u00b1 {row['f1_std']:.4f}"
        acc_str = f"{row['in_acc_mean']:.4f} \u00b1 {row['in_acc_std']:.4f}"
        fp_str = f"{row['fp_mean']:.1f} \u00b1 {row['fp_std']:.1f}"
        print(f"{row['threshold']:<16.2f} | {rec_str:<18} | {prec_str:<18} | {f1_str:<18} | {acc_str:<18} | {fp_str:<18}")

    # 2. Compute 95% Bootstrap Confidence Intervals for Exemplar Cosine & Learned Gate
    print("\n" + "=" * 90)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (VALIDATION SPLIT, 1000 resamples)")
    print("=" * 90)

    # Re-encode for Exemplar Cosine baseline
    train_texts = [x["text"] for x in splits["train"]]
    train_labels = [x["label"] for x in splits["train"]]
    exemplar_model = ExemplarCosineBaseline()
    exemplar_model.fit(train_texts, train_labels)
    exemplar_scores = exemplar_model.predict_oos_score(val_texts)
    th_exemplar = 0.5504

    ci_exemplar = compute_bootstrap_confidence_intervals(
        exemplar_scores, val_labels, threshold=th_exemplar, n_bootstraps=1000, seed=42
    )
    m_exemplar = compute_binary_metrics_at_threshold(exemplar_scores, val_labels, threshold=th_exemplar)

    print(f"Exemplar Cosine Baseline (tau={th_exemplar}):")
    print(f"  OOS F1:       {m_exemplar['oos_f1']:.4f} [{ci_exemplar['oos_f1'][0]:.4f}, {ci_exemplar['oos_f1'][1]:.4f}]")
    print(f"  OOS Recall:   {m_exemplar['oos_recall']:.4f} [{ci_exemplar['oos_recall'][0]:.4f}, {ci_exemplar['oos_recall'][1]:.4f}]")
    print(f"  OOS Prec:     {m_exemplar['oos_precision']:.4f} [{ci_exemplar['oos_precision'][0]:.4f}, {ci_exemplar['oos_precision'][1]:.4f}]")
    print(f"  In-Scope Acc: {m_exemplar['in_scope_acceptance']:.4f} [{ci_exemplar['in_scope_acceptance'][0]:.4f}, {ci_exemplar['in_scope_acceptance'][1]:.4f}]")
    print(f"  FP: {m_exemplar['fp']}, FN: {m_exemplar['fn']}")

    learned_cis = {}
    for s in seeds:
        th = predictors[s].threshold
        ci = compute_bootstrap_confidence_intervals(
            scores_dict[s], val_labels, threshold=th, n_bootstraps=1000, seed=42
        )
        m = compute_binary_metrics_at_threshold(scores_dict[s], val_labels, threshold=th)
        learned_cis[s] = {"metrics": m, "ci": ci, "threshold": th}
        print(f"\nLearned Gate Seed {s} (tau={th:.4f}):")
        print(f"  OOS F1:       {m['oos_f1']:.4f} [{ci['oos_f1'][0]:.4f}, {ci['oos_f1'][1]:.4f}]")
        print(f"  OOS Recall:   {m['oos_recall']:.4f} [{ci['oos_recall'][0]:.4f}, {ci['oos_recall'][1]:.4f}]")
        print(f"  OOS Prec:     {m['oos_precision']:.4f} [{ci['oos_precision'][0]:.4f}, {ci['oos_precision'][1]:.4f}]")
        print(f"  In-Scope Acc: {m['in_scope_acceptance']:.4f} [{ci['in_scope_acceptance'][0]:.4f}, {ci['in_scope_acceptance'][1]:.4f}]")
        print(f"  FP: {m['fp']}, FN: {m['fn']}")

    # Save output to json for reference
    out_file = Path("results/val_sweep_and_ci.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "threshold_sweep": sweep_results,
            "exemplar_baseline": {"metrics": m_exemplar, "ci": ci_exemplar, "threshold": th_exemplar},
            "learned_gate_seeds": learned_cis,
        }, f, indent=2)
    print(f"\nSaved raw sweep and bootstrap data to {out_file}")


if __name__ == "__main__":
    main()
