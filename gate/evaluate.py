"""Evaluation and benchmarking module for Scope Gate models.

Evaluates learned scope gate checkpoints and baseline models on evaluation sets
using decision thresholds tuned strictly on the validation split.

Key capabilities:
- Zero threshold leakage: thresholds chosen strictly on validation
- Expected Calibration Error (ECE) and reliability diagrams
- 95% bootstrap confidence intervals (>= 1000 iterations)
- Multi-seed reporting (Seed 42, 1337, 2026 and mean \u00b1 std)
- Per-method and per-seed results formatting
- Raw predictions export for error analysis
- Campus dataset verification (refuses to run without matching SHA-256 hash file)
- Campus author subset analysis (author = 'human' vs 'ai_draft')
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gate.data import (
    load_clinc_data,
    LABEL_IN_SCOPE,
    LABEL_OOS,
)
from gate.baselines import (
    BaseGateBaseline,
    ExemplarCosineBaseline,
    CentroidCosineBaseline,
    TfidfLogisticRegressionBaseline,
    FrozenEmbeddingLogisticRegressionBaseline,
    compute_binary_metrics_at_threshold,
    find_optimal_threshold,
)
from gate.predict import ScopeGatePredictor

logger = logging.getLogger(__name__)

DEFAULT_HASH_FILE = Path("results/campus_hash.sha256")
DEFAULT_CAMPUS_CSV = Path("data/campus_eval/campus_questions.csv")


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE) for binary predictions.

    Args:
        probs: Model predicted probabilities for the positive class (OOS, class 0) or max confidence.
               Here we calibrate OOS probability against true OOS indicator (1 if true OOS else 0).
        labels: Ground truth binary labels (LABEL_IN_SCOPE=1, LABEL_OOS=0).
        num_bins: Number of equal-width confidence bins in [0, 1].

    Returns:
        Tuple of (ece_value, bin_details_dict)
    """
    # Positive class for calibration is OOS (ground truth == LABEL_OOS)
    true_oos = (labels == LABEL_OOS).astype(int)
    probs = np.clip(probs, 0.0, 1.0)
    n = len(labels)

    if n == 0:
        return 0.0, {"bin_confs": [], "bin_accs": [], "bin_counts": []}

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = int(np.sum(in_bin))
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_conf = float(np.mean(probs[in_bin]))
            bin_acc = float(np.mean(true_oos[in_bin]))
            bin_confs.append(bin_conf)
            bin_accs.append(bin_acc)
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
        else:
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_accs.append(0.0)

    return float(ece), {
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def plot_reliability_diagram(
    bin_details: Dict[str, Any],
    ece: float,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Generate and save a publication-quality reliability diagram.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_confs = np.array(bin_details["bin_confs"])
    bin_accs = np.array(bin_details["bin_accs"])
    bin_counts = np.array(bin_details["bin_counts"])
    num_bins = len(bin_counts)
    bin_width = 1.0 / num_bins

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    bin_centers = np.linspace(bin_width / 2, 1.0 - bin_width / 2, num_bins)

    # Plot bars with gaps
    for center, acc, conf, count in zip(bin_centers, bin_accs, bin_confs, bin_counts):
        if count > 0:
            ax1.bar(
                center,
                acc,
                width=bin_width * 0.9,
                alpha=0.6,
                color="#2b5c8f",
                edgecolor="black",
                linewidth=0.8,
            )
            # Gap bar
            ax1.bar(
                center,
                abs(acc - conf),
                bottom=min(acc, conf),
                width=bin_width * 0.9,
                alpha=0.3,
                color="#d9534f",
                hatch="//",
            )

    ax1.set_ylabel("Empirical Accuracy P(OOS)")
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title(f"Reliability Diagram: {model_name}\nECE = {ece:.4f}")
    ax1.legend(["Perfect Calibration", "Empirical Accuracy", "Calibration Gap"], loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Confidence histogram
    total_samples = max(1, int(np.sum(bin_counts)))
    proportions = bin_counts / total_samples
    ax2.bar(
        bin_centers,
        proportions,
        width=bin_width * 0.9,
        color="#6c757d",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_xlabel("Predicted Confidence P(OOS)")
    ax2.set_ylabel("% Samples")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_bootstrap_confidence_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute empirical percentile bootstrap confidence intervals by resampling items with replacement.

    Args:
        scores: 1D array of OOS probability scores.
        labels: 1D array of ground truth labels (LABEL_IN_SCOPE=1, LABEL_OOS=0).
        threshold: Pre-tuned validation decision threshold.
        n_bootstraps: Number of bootstrap iterations (>= 1000).
        ci: Confidence interval nominal width (default 0.95).
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        Dict mapping metric name to (ci_lower, ci_upper) tuple.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n == 0:
        return {}

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    boot_f1: List[float] = []
    boot_recall: List[float] = []
    boot_precision: List[float] = []
    boot_in_scope_acc: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        b_scores = scores[boot_idx]
        b_labels = labels[boot_idx]

        m = compute_binary_metrics_at_threshold(b_scores, b_labels, threshold=threshold)
        boot_f1.append(m["oos_f1"])
        boot_recall.append(m["oos_recall"])
        boot_precision.append(m["oos_precision"])
        boot_in_scope_acc.append(m["in_scope_acceptance"])

    return {
        "oos_f1": (
            float(np.percentile(boot_f1, lower_pct)),
            float(np.percentile(boot_f1, upper_pct)),
        ),
        "oos_recall": (
            float(np.percentile(boot_recall, lower_pct)),
            float(np.percentile(boot_recall, upper_pct)),
        ),
        "oos_precision": (
            float(np.percentile(boot_precision, lower_pct)),
            float(np.percentile(boot_precision, upper_pct)),
        ),
        "in_scope_acceptance": (
            float(np.percentile(boot_in_scope_acc, lower_pct)),
            float(np.percentile(boot_in_scope_acc, upper_pct)),
        ),
    }


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_campus_hash(
    campus_csv_path: Union[str, Path] = DEFAULT_CAMPUS_CSV,
    hash_file_path: Union[str, Path] = DEFAULT_HASH_FILE,
) -> str:
    """
    Verify that campus_questions.csv matches its frozen SHA-256 checksum.

    Refuses execution and raises an error if:
    1. The hash file does not exist.
    2. The computed SHA-256 does not match the frozen hash.
    """
    campus_csv = Path(campus_csv_path)
    hash_file = Path(hash_file_path)

    if not campus_csv.exists():
        raise FileNotFoundError(f"Campus dataset file not found: {campus_csv}")

    if not hash_file.exists():
        raise FileNotFoundError(
            f"Campus hash file not found at: {hash_file}. "
            "Refusing to run evaluation until the campus dataset is frozen and its SHA-256 hash is saved."
        )

    expected_hash = hash_file.read_text(encoding="utf-8").strip()
    actual_hash = compute_file_sha256(campus_csv)

    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"Campus dataset integrity verification failed!\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Actual SHA-256:   {actual_hash}\n"
            f"Dataset has been modified after freezing. Refusing to run."
        )

    logger.info(f"Campus dataset verified against frozen SHA-256: {actual_hash[:16]}...")
    return actual_hash


def load_campus_dataset(
    campus_csv_path: Union[str, Path] = DEFAULT_CAMPUS_CSV,
    hash_file_path: Optional[Union[str, Path]] = DEFAULT_HASH_FILE,
    verify_hash: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load campus_questions.csv with integrity and schema validation.
    """
    campus_csv = Path(campus_csv_path)
    if verify_hash and hash_file_path is not None:
        verify_campus_hash(campus_csv, hash_file_path)

    rows: List[Dict[str, Any]] = []
    with open(campus_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl_str = row["label"].strip().lower()
            binary_label = LABEL_IN_SCOPE if lbl_str == "in" else LABEL_OOS
            rows.append({
                "question": row["question"].strip(),
                "label": binary_label,
                "source_doc": row.get("source_doc", "").strip(),
                "lang": row.get("lang", "en").strip().lower(),
                "difficulty": row.get("difficulty", "easy").strip().lower(),
                "author": row.get("author", "ai_draft").strip().lower(),
                "notes": row.get("notes", "").strip(),
            })

    return rows


def evaluate_single_method(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
    num_ece_bins: int = 10,
    save_reliability_diagram_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full evaluation of a single method on a dataset.
    """
    labels = np.array(labels, dtype=int)
    scores = np.array(scores, dtype=np.float32)

    # Point metrics
    m = compute_binary_metrics_at_threshold(scores, labels, threshold=threshold)

    # Calibration
    ece, bin_details = compute_ece(scores, labels, num_bins=num_ece_bins)
    if save_reliability_diagram_path is not None:
        plot_reliability_diagram(bin_details, ece, model_name=name, output_path=save_reliability_diagram_path)

    # 95% Bootstrap Confidence Intervals
    boot_ci = compute_bootstrap_confidence_intervals(
        scores, labels, threshold=threshold, n_bootstraps=n_bootstraps, seed=seed
    )

    return {
        "method": name,
        "threshold": float(threshold),
        "total_samples": len(labels),
        "oos_samples": int(np.sum(labels == LABEL_OOS)),
        "in_scope_samples": int(np.sum(labels == LABEL_IN_SCOPE)),
        "metrics": {
            "oos_f1": m["oos_f1"],
            "oos_recall": m["oos_recall"],
            "oos_precision": m["oos_precision"],
            "in_scope_acceptance": m["in_scope_acceptance"],
            "ece": ece,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        },
        "confidence_intervals_95": boot_ci,
        "calibration": {
            "ece": ece,
            "bin_details": bin_details,
        },
    }


def format_metric_with_ci(value: float, ci: Optional[Tuple[float, float]] = None) -> str:
    """Format metric value with optional [lower, upper] 95% CI string."""
    if ci is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def print_results_table(
    eval_results: List[Dict[str, Any]],
    table_title: str = "Evaluation Results",
    include_ci: bool = True,
) -> None:
    """Print markdown and formatted ASCII tables for results."""
    print(f"\n### {table_title}\n")
    headers = ["Method", "Threshold", "OOS F1", "OOS Recall", "OOS Precision", "In-Scope Acceptance", "ECE"]
    
    col_widths = [42, 11, 24 if include_ci else 10, 24 if include_ci else 10, 24 if include_ci else 10, 24 if include_ci else 14, 10]
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)

    print(header_str)
    print(sep_str)

    for r in eval_results:
        m = r["metrics"]
        ci = r.get("confidence_intervals_95", {})
        
        f1_str = format_metric_with_ci(m["oos_f1"], ci.get("oos_f1") if include_ci else None)
        rec_str = format_metric_with_ci(m["oos_recall"], ci.get("oos_recall") if include_ci else None)
        prec_str = format_metric_with_ci(m["oos_precision"], ci.get("oos_precision") if include_ci else None)
        acc_str = format_metric_with_ci(m["in_scope_acceptance"], ci.get("in_scope_acceptance") if include_ci else None)
        ece_str = f"{m['ece']:.4f}"

        row_str = (
            f"{r['method']:<42} | "
            f"{r['threshold']:<11.4f} | "
            f"{f1_str:<{col_widths[2]}} | "
            f"{rec_str:<{col_widths[3]}} | "
            f"{prec_str:<{col_widths[4]}} | "
            f"{acc_str:<{col_widths[5]}} | "
            f"{ece_str:<10}"
        )
        print(row_str)

    print(sep_str + "\n")


def compute_aggregate_seed_stats(
    seed_evals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute mean and std across multiple seed evaluations."""
    f1s = [e["metrics"]["oos_f1"] for e in seed_evals]
    recs = [e["metrics"]["oos_recall"] for e in seed_evals]
    precs = [e["metrics"]["oos_precision"] for e in seed_evals]
    accs = [e["metrics"]["in_scope_acceptance"] for e in seed_evals]
    eces = [e["metrics"]["ece"] for e in seed_evals]
    ths = [e["threshold"] for e in seed_evals]

    return {
        "method": "Learned Gate (Mean \u00b1 Std across seeds)",
        "threshold_mean": float(np.mean(ths)),
        "threshold_std": float(np.std(ths)),
        "oos_f1_mean": float(np.mean(f1s)),
        "oos_f1_std": float(np.std(f1s)),
        "oos_recall_mean": float(np.mean(recs)),
        "oos_recall_std": float(np.std(recs)),
        "oos_precision_mean": float(np.mean(precs)),
        "oos_precision_std": float(np.std(precs)),
        "in_scope_acceptance_mean": float(np.mean(accs)),
        "in_scope_acceptance_std": float(np.std(accs)),
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
    }


def fit_and_tune_baselines(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
) -> List[Tuple[str, Any, float]]:
    """
    Fit the 4 baselines on train split and tune decision threshold strictly on validation split.
    Returns list of tuples: (model_name, fitted_model, tuned_val_threshold).
    """
    logger.info("Fitting and tuning baselines on train and validation...")
    val_labels_arr = np.array(val_labels, dtype=int)
    baselines: List[Tuple[str, Any, float]] = []

    # 1. TF-IDF + Logistic Regression
    tfidf = TfidfLogisticRegressionBaseline(seed=42)
    tfidf.fit(train_texts, train_labels)
    th_tfidf = tfidf.tune_threshold_on_validation(val_texts, val_labels, criterion="max_f1")
    baselines.append(("TF-IDF + Logistic Regression", tfidf, th_tfidf))

    # 2. Centroid Cosine Similarity
    centroid = CentroidCosineBaseline()
    centroid.fit(train_texts, train_labels)
    th_centroid = centroid.tune_threshold_on_validation(val_texts, val_labels, criterion="max_f1")
    baselines.append(("Centroid Cosine Similarity", centroid, th_centroid))

    # 3. Exemplar Cosine Similarity
    exemplar = ExemplarCosineBaseline()
    exemplar.fit(train_texts, train_labels)
    th_exemplar = exemplar.tune_threshold_on_validation(val_texts, val_labels, criterion="max_f1")
    baselines.append(("Exemplar Cosine Similarity (Nearest Neighbor)", exemplar, th_exemplar))

    # 4. Frozen Embedding + Logistic Regression
    frozen = FrozenEmbeddingLogisticRegressionBaseline(seed=42)
    frozen.fit(train_texts, train_labels)
    th_frozen = frozen.tune_threshold_on_validation(val_texts, val_labels, criterion="max_f1")
    baselines.append(("Frozen Embedding + Logistic Regression", frozen, th_frozen))

    return baselines


def load_learned_gate_models(
    checkpoint_dir: Path = Path("results/checkpoints"),
    seeds: List[int] = (42, 1337, 2026),
) -> List[Tuple[str, ScopeGatePredictor, float]]:
    """
    Load trained learned-gate checkpoints and retrieve validation-tuned thresholds.
    """
    models: List[Tuple[str, ScopeGatePredictor, float]] = []
    for s in seeds:
        ckpt_path = checkpoint_dir / f"scope_gate_seed_{s}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for seed {s}: {ckpt_path}")
        predictor = ScopeGatePredictor(ckpt_path)
        th = predictor.threshold
        models.append((f"Learned Gate (Seed {s})", predictor, th))
    return models


def evaluate_all_methods_on_split(
    methods: List[Tuple[str, Any, float]],
    eval_texts: List[str],
    eval_labels: List[int],
    split_name: str,
    output_dir: Path = Path("results"),
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate all methods on a given evaluation split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagrams_dir = output_dir / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    labels_arr = np.array(eval_labels, dtype=int)
    results: List[Dict[str, Any]] = []
    raw_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for name, model, th in methods:
        logger.info(f"Evaluating {name} on {split_name} (threshold={th:.4f})...")
        scores = model.predict_oos_score(eval_texts)

        clean_slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        diag_path = diagrams_dir / f"reliability_{clean_slug}_{split_name.lower()}.png"

        eval_res = evaluate_single_method(
            name=name,
            scores=scores,
            labels=labels_arr,
            threshold=th,
            n_bootstraps=n_bootstraps,
            save_reliability_diagram_path=diag_path,
        )
        results.append(eval_res)

        preds = (scores >= th).astype(int)
        # Note: binary_label: 0=OOS, 1=In-Scope
        pred_labels = np.where(preds == 1, LABEL_OOS, LABEL_IN_SCOPE)
        raw_predictions[name] = [
            {
                "query": text,
                "true_label": int(lbl),
                "true_name": "oos" if lbl == LABEL_OOS else "in_scope",
                "predicted_label": int(p_lbl),
                "predicted_name": "oos" if p_lbl == LABEL_OOS else "in_scope",
                "oos_score": float(score),
                "threshold": float(th),
                "is_correct": bool(lbl == p_lbl),
            }
            for text, lbl, p_lbl, score in zip(eval_texts, eval_labels, pred_labels, scores)
        ]

    # Compute aggregate seed stats
    seed_evals = [r for r in results if "Learned Gate (Seed" in r["method"]]
    aggregate_seed_stats = None
    if seed_evals:
        aggregate_seed_stats = compute_aggregate_seed_stats(seed_evals)

    # Save raw predictions
    preds_file = output_dir / f"{split_name.lower()}_predictions.json"
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(raw_predictions, f, indent=2)
    logger.info(f"Saved raw predictions to: {preds_file}")

    return {
        "split_name": split_name,
        "results": results,
        "aggregate_seed_stats": aggregate_seed_stats,
        "raw_predictions_path": str(preds_file),
    }


def evaluate_campus_breakdown(
    methods: List[Tuple[str, Any, float]],
    campus_rows: List[Dict[str, Any]],
    output_dir: Path = Path("results"),
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate campus evaluation set with strict author subset breakdown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_texts = [r["question"] for r in campus_rows]
    all_labels = [r["label"] for r in campus_rows]

    # Evaluate overall campus set
    overall_res = evaluate_all_methods_on_split(
        methods=methods,
        eval_texts=all_texts,
        eval_labels=all_labels,
        split_name="campus_overall",
        output_dir=output_dir,
        n_bootstraps=n_bootstraps,
    )

    # Evaluate author subsets
    subsets = {
        "campus_human": [r for r in campus_rows if r["author"] == "human"],
        "campus_ai_draft": [r for r in campus_rows if r["author"] == "ai_draft"],
        "campus_en": [r for r in campus_rows if r["lang"] == "en"],
        "campus_de": [r for r in campus_rows if r["lang"] == "de"],
        "campus_easy": [r for r in campus_rows if r["difficulty"] == "easy"],
        "campus_hard": [r for r in campus_rows if r["difficulty"] == "hard"],
    }

    subset_results: Dict[str, Any] = {}
    for subset_name, subset_rows in subsets.items():
        if not subset_rows:
            logger.info(f"Subset {subset_name} is empty (0 rows); skipping.")
            continue
        logger.info(f"\n--- Evaluating subset: {subset_name} ({len(subset_rows)} rows) ---")
        sub_texts = [r["question"] for r in subset_rows]
        sub_labels = [r["label"] for r in subset_rows]
        sub_res = evaluate_all_methods_on_split(
            methods=methods,
            eval_texts=sub_texts,
            eval_labels=sub_labels,
            split_name=subset_name,
            output_dir=output_dir,
            n_bootstraps=n_bootstraps,
        )
        subset_results[subset_name] = sub_res

    return {
        "overall": overall_res,
        "subsets": subset_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scope Gate on test or validation splits and campus set.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="CLINC split to evaluate. Note: use validation until campus set is frozen.")
    parser.add_argument("--campus", action="store_true", help="Evaluate campus questions dataset.")
    parser.add_argument("--campus-csv", type=str, default=str(DEFAULT_CAMPUS_CSV), help="Path to campus questions CSV.")
    parser.add_argument("--campus-hash", type=str, default=str(DEFAULT_HASH_FILE), help="Path to campus hash file.")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap iterations (>= 1000).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation results.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    splits = load_clinc_data(use_cache=True, dedup_leaks=True)

    train_texts = [x["text"] for x in splits["train"]]
    train_labels = [x["label"] for x in splits["train"]]

    val_texts = [x["text"] for x in splits["validation"]]
    val_labels = [x["label"] for x in splits["validation"]]

    # Fit and tune baselines on validation
    baselines = fit_and_tune_baselines(train_texts, train_labels, val_texts, val_labels)

    # Load learned gate checkpoints (thresholds loaded from validation metadata)
    learned_models = load_learned_gate_models()
    all_methods = baselines + learned_models

    out_dir = Path(args.output_dir)

    # 1. Evaluate CLINC target split
    target_texts = [x["text"] for x in splits[args.split]]
    target_labels = [x["label"] for x in splits[args.split]]

    clinc_res = evaluate_all_methods_on_split(
        methods=all_methods,
        eval_texts=target_texts,
        eval_labels=target_labels,
        split_name=f"clinc_{args.split}",
        output_dir=out_dir,
        n_bootstraps=args.bootstraps,
    )
    print_results_table(clinc_res["results"], table_title=f"CLINC {args.split.upper()} Evaluation Results (95% Bootstrap CI)")

    if clinc_res["aggregate_seed_stats"]:
        s = clinc_res["aggregate_seed_stats"]
        print(f"\n{s['method']}:")
        print(f"  OOS F1:               {s['oos_f1_mean']:.4f} \u00b1 {s['oos_f1_std']:.4f}")
        print(f"  OOS Recall:           {s['oos_recall_mean']:.4f} \u00b1 {s['oos_recall_std']:.4f}")
        print(f"  OOS Precision:        {s['oos_precision_mean']:.4f} \u00b1 {s['oos_precision_std']:.4f}")
        print(f"  In-Scope Acceptance:  {s['in_scope_acceptance_mean']:.4f} \u00b1 {s['in_scope_acceptance_std']:.4f}")
        print(f"  ECE:                  {s['ece_mean']:.4f} \u00b1 {s['ece_std']:.4f}")

    # 2. Evaluate Campus Dataset if requested
    if args.campus:
        campus_rows = load_campus_dataset(
            campus_csv_path=args.campus_csv,
            hash_file_path=args.campus_hash,
            verify_hash=True,
        )
        campus_res = evaluate_campus_breakdown(
            methods=all_methods,
            campus_rows=campus_rows,
            output_dir=out_dir,
            n_bootstraps=args.bootstraps,
        )
        print_results_table(
            campus_res["overall"]["results"],
            table_title="Campus Evaluation Results (Overall, 95% Bootstrap CI)",
        )

        for sub_name, sub_data in campus_res["subsets"].items():
            print_results_table(
                sub_data["results"],
                table_title=f"Campus Subset: {sub_name} (95% Bootstrap CI)",
            )


if __name__ == "__main__":
    main()
