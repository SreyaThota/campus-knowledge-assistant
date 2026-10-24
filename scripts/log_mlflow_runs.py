"""MLflow experiment logging and summary generation for Scope Gate training runs.

Reads the 3-seed training curves from results/training_curves.json and logs:
- Hyperparameters (encoder backbone, lr, batch_size, epochs, loss function, weight_decay, warmup)
- Random seeds (42, 1337, 2026)
- Per-epoch training and validation loss, threshold, F1, recall, precision, in-scope acceptance
- Best validation metrics and tuned decision thresholds
- Aggregate multi-seed statistics (mean \u00b1 std)

Outputs a concise, committable summary report at results/mlflow_summary.md.
VALIDATION DATA ONLY.
"""

import json
from pathlib import Path
from typing import Dict, Any
import mlflow

TRAINING_CURVES_PATH = Path("results/training_curves.json")
SUMMARY_OUTPUT_PATH = Path("results/mlflow_summary.md")
EXPERIMENT_NAME = "learned-scope-gate"


def log_curves_to_mlflow(
    curves_path: Path = TRAINING_CURVES_PATH,
    experiment_name: str = EXPERIMENT_NAME,
) -> Dict[str, Any]:
    """Log the 3-seed training curves to MLflow without re-running training."""
    if not curves_path.exists():
        raise FileNotFoundError(f"Training curves file not found: {curves_path}")

    with open(curves_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mlflow.set_experiment(experiment_name)
    logged_runs = []

    # 1. Parent run for the multi-seed experiment
    with mlflow.start_run(run_name="learned_gate_3seed_aggregate") as parent_run:
        parent_id = parent_run.info.run_id
        mlflow.log_params({
            "seeds": str(data["seeds"]),
            "num_epochs": data["num_epochs"],
            "batch_size": data["batch_size"],
            "learning_rate": data["learning_rate"],
            "device": data["device"],
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "loss_function": "CrossEntropyLoss(class_weighted)",
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "evaluation_split": "validation_only",
        })

        agg = data["aggregate_validation_metrics"]
        mlflow.log_metrics({
            "val_oos_f1_mean": agg["val_oos_f1_mean"],
            "val_oos_f1_std": agg["val_oos_f1_std"],
            "val_oos_recall_mean": agg["val_oos_recall_mean"],
            "val_oos_recall_std": agg["val_oos_recall_std"],
            "val_oos_precision_mean": agg["val_oos_precision_mean"],
            "val_oos_precision_std": agg["val_oos_precision_std"],
            "val_in_scope_acc_mean": agg["val_in_scope_acceptance_mean"],
            "val_in_scope_acc_std": agg["val_in_scope_acceptance_std"],
            "val_threshold_mean": agg["val_threshold_mean"],
            "val_threshold_std": agg["val_threshold_std"],
        })

        # 2. Child runs for each individual seed
        for run_info in data["seed_runs"]:
            seed = run_info["seed"]
            with mlflow.start_run(run_name=f"learned_gate_seed_{seed}", nested=True) as child_run:
                child_id = child_run.info.run_id
                mlflow.log_params({
                    "seed": seed,
                    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                    "num_epochs": data["num_epochs"],
                    "batch_size": data["batch_size"],
                    "learning_rate": data["learning_rate"],
                    "best_epoch": run_info["best_epoch"],
                    "checkpoint_path": run_info["checkpoint_path"],
                    "total_duration_sec": run_info["total_duration_sec"],
                })

                # Log per-epoch history
                for epoch_data in run_info["epochs"]:
                    ep = epoch_data["epoch"]
                    mlflow.log_metrics({
                        "train_loss": epoch_data["train_loss"],
                        "val_loss": epoch_data["val_loss"],
                        "val_threshold": epoch_data["val_threshold"],
                        "oos_f1": epoch_data["oos_f1"],
                        "oos_recall": epoch_data["oos_recall"],
                        "oos_precision": epoch_data["oos_precision"],
                        "in_scope_acceptance": epoch_data["in_scope_acceptance"],
                    }, step=ep)

                # Log final tuned validation metrics
                best_m = run_info["best_val_metrics"]
                mlflow.log_metrics({
                    "best_val_threshold": run_info["best_val_threshold"],
                    "best_val_oos_f1": best_m["oos_f1"],
                    "best_val_oos_recall": best_m["oos_recall"],
                    "best_val_oos_precision": best_m["oos_precision"],
                    "best_val_in_scope_acceptance": best_m["in_scope_acceptance"],
                    "best_val_tp": best_m["tp"],
                    "best_val_fp": best_m["fp"],
                    "best_val_fn": best_m["fn"],
                    "best_val_tn": best_m["tn"],
                })

                logged_runs.append({
                    "seed": seed,
                    "run_id": child_id,
                    "best_epoch": run_info["best_epoch"],
                    "best_val_threshold": run_info["best_val_threshold"],
                    "best_val_metrics": best_m,
                    "duration_sec": run_info["total_duration_sec"],
                })

    return {
        "parent_run_id": parent_id,
        "experiment_name": experiment_name,
        "aggregate": agg,
        "runs": logged_runs,
    }


def write_summary_markdown(
    log_info: Dict[str, Any],
    output_path: Path = SUMMARY_OUTPUT_PATH,
) -> None:
    """Generate concise commit-ready MLflow tracking summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg = log_info["aggregate"]

    md = f"""# MLflow Experiment Tracking Summary: Learned Scope Gate

> **NOTICE: VALIDATION SPLIT TRACKING ONLY**
>
> All tracked runs, metrics, and thresholds logged below reflect tuning exclusively on the **CLINC150 validation split** ($N = 3,100$: $3,000$ in-scope, $100$ out-of-scope).
> **No test split data or campus evaluation data was used.**

---

## Experiment Metadata

- **Experiment Name:** `{log_info['experiment_name']}`
- **Parent Run ID:** `{log_info['parent_run_id']}`
- **Model Backbone:** `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters)
- **Classifier Head:** Mean Pooling \u2192 Dropout(0.2) \u2192 Linear(384 \u2192 2)
- **Loss Function:** Class-Weighted CrossEntropyLoss (inverse frequency: $w_\\text{{oos}} \\approx 31.1$, $w_\\text{{in}} \\approx 0.51$)
- **Optimization:** AdamW (lr = 3e-5, weight decay = 0.01, linear warmup ratio = 0.1, max grad norm = 1.0)
- **Hardware Device:** CPU execution

---

## Logged Runs by Seed

| Run Name | Seed | MLflow Run ID | Best Epoch | Tuned Threshold ($\\tau$) | Val OOS $F_1$ | Val OOS Recall | Val OOS Precision | Val In-Scope Acc | Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for r in log_info["runs"]:
        m = r["best_val_metrics"]
        md += (
            f"| `learned_gate_seed_{r['seed']}` | {r['seed']} | `{r['run_id'][:8]}...` | "
            f"{r['best_epoch']} | {r['best_val_threshold']:.4f} | {m['oos_f1']:.4f} | "
            f"{m['oos_recall']:.4f} | {m['oos_precision']:.4f} | {m['in_scope_acceptance']:.4f} | "
            f"{r['duration_sec']:.1f} |\n"
        )

    md += f"""
---

## Multi-Seed Validation Aggregate

- **Validation OOS $F_1$:** {agg['val_oos_f1_mean']:.4f} \u00b1 {agg['val_oos_f1_std']:.4f}
- **Validation OOS Recall:** {agg['val_oos_recall_mean']:.4f} \u00b1 {agg['val_oos_recall_std']:.4f}
- **Validation OOS Precision:** {agg['val_oos_precision_mean']:.4f} \u00b1 {agg['val_oos_precision_std']:.4f}
- **Validation In-Scope Acceptance:** {agg['val_in_scope_acceptance_mean']:.4f} \u00b1 {agg['val_in_scope_acceptance_std']:.4f}
- **Optimal Decision Threshold ($\\tau$):** {agg['val_threshold_mean']:.4f} \u00b1 {agg['val_threshold_std']:.4f}

---

## How to Inspect Local MLflow Dashboard

To view the interactive MLflow UI with complete parameter diffs, epoch learning curves, and artifacts:
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```
"""

    output_path.write_text(md, encoding="utf-8")
    print(f"Saved MLflow summary to: {output_path}")


def main():
    info = log_curves_to_mlflow()
    write_summary_markdown(info)


if __name__ == "__main__":
    main()
