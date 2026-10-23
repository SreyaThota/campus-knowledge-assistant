"""Hand-written PyTorch training loop for Learned Scope Gate.

Key constraints:
- Strictly hand-written loop (torch.nn.Module, DataLoader, AdamW, linear warmup + decay,
  gradient clipping, early stopping on validation).
- Zero high-level wrappers (no Hugging Face Trainer).
- Class-weighted CrossEntropyLoss to address severe class imbalance (~60:1).
- 3 fixed seeds (42, 1337, 2026).
- Thresholds and early stopping evaluated exclusively on the validation split.
- Training curves and checkpoints saved under results/.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from gate.data import load_clinc_data, LABEL_IN_SCOPE, LABEL_OOS
from gate.model import ScopeGateClassifier, DEFAULT_ENCODER
from gate.baselines import compute_binary_metrics_at_threshold, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_tensor_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer: AutoTokenizer,
    max_length: int = 36,
) -> TensorDataset:
    """
    Tokenize all texts upfront into static tensors for maximum throughput on CPU.
    """
    logger.info(f"Tokenizing {len(texts)} samples (max_length={max_length})...")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels_tensor)


def evaluate_on_validation(
    model: ScopeGateClassifier,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute validation loss and collect OOS probability predictions across validation split.
    """
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    all_oos_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_val_loss += loss.item()
            val_steps += 1

            probs = torch.softmax(logits, dim=-1)
            oos_probs = probs[:, LABEL_OOS].cpu().numpy()
            all_oos_probs.extend(oos_probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_val_loss = total_val_loss / max(1, val_steps)
    return avg_val_loss, np.array(all_oos_probs, dtype=np.float32), np.array(all_labels, dtype=int)


def train_single_seed(
    seed: int,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    num_epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: Path = Path("results/checkpoints"),
) -> Dict[str, Any]:
    """
    Train a single model instance using a hand-written training loop.
    """
    logger.info(f"\n==================== Starting Training (Seed {seed}) ====================")
    set_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Compute class weights from training targets
    train_labels = train_dataset.tensors[2].numpy()
    n_total = len(train_labels)
    n_oos = int(np.sum(train_labels == LABEL_OOS))
    n_in = int(np.sum(train_labels == LABEL_IN_SCOPE))

    w_oos = float(n_total / (2.0 * max(1, n_oos)))
    w_in = float(n_total / (2.0 * max(1, n_in)))
    logger.info(f"Class counts - In-scope: {n_in}, OOS: {n_oos}. Balanced loss weights: [OOS={w_oos:.2f}, InScope={w_in:.4f}]")

    class_weights = torch.tensor([w_oos, w_in], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = ScopeGateClassifier(encoder_name=DEFAULT_ENCODER).to(device)

    # Differential weight decay for AdamW
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    epoch_logs: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    best_epoch = 1

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"scope_gate_seed_{seed}.pt"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)} | "
                    f"Train Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation evaluation
        avg_val_loss, val_oos_probs, val_labels = evaluate_on_validation(
            model, val_loader, criterion, device
        )

        # Find threshold that maximizes OOS F1 on validation split
        val_th = find_optimal_threshold(val_oos_probs, val_labels, criterion="max_f1")
        val_metrics = compute_binary_metrics_at_threshold(val_oos_probs, val_labels, val_th)

        epoch_duration = time.time() - epoch_start
        logger.info(
            f"--> Epoch {epoch} Complete ({epoch_duration:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val OOS F1: {val_metrics['oos_f1']:.4f} | "
            f"Val OOS Recall: {val_metrics['oos_recall']:.4f} | "
            f"In-Scope Acceptance: {val_metrics['in_scope_acceptance']:.4f} | "
            f"Tuned Threshold: {val_th:.3f}"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_threshold": round(val_th, 4),
            "oos_f1": round(val_metrics["oos_f1"], 4),
            "oos_recall": round(val_metrics["oos_recall"], 4),
            "oos_precision": round(val_metrics["oos_precision"], 4),
            "in_scope_acceptance": round(val_metrics["in_scope_acceptance"], 4),
            "duration_sec": round(epoch_duration, 1),
        })

        # Save checkpoint if best validation F1
        if val_metrics["oos_f1"] > best_val_f1:
            best_val_f1 = val_metrics["oos_f1"]
            best_threshold = val_th
            best_metrics = val_metrics
            best_epoch = epoch

            metadata = {
                "seed": seed,
                "best_epoch": best_epoch,
                "val_threshold": best_threshold,
                "val_metrics": best_metrics,
                "encoder_name": DEFAULT_ENCODER,
                "class_weights": [w_oos, w_in],
            }
            model.save_checkpoint(best_checkpoint_path, metadata=metadata)
            logger.info(f"*** New best checkpoint saved for seed {seed} (F1: {best_val_f1:.4f}) -> {best_checkpoint_path} ***")

    total_duration = time.time() - start_time
    logger.info(f"Training for seed {seed} finished in {total_duration:.1f}s. Best Epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}\n")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_threshold": round(best_threshold, 4),
        "best_val_metrics": best_metrics,
        "checkpoint_path": str(best_checkpoint_path),
        "total_duration_sec": round(total_duration, 1),
        "epochs": epoch_logs,
    }


def run_training_multi_seeds(
    seeds: List[int] = (42, 1337, 2026),
    num_epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Train across multiple fixed seeds, save curves and validation summary.
    """
    device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"

    splits = load_clinc_data(use_cache=True, dedup_leaks=True)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER)

    train_texts = [x["text"] for x in splits["train"]]
    train_labels = [x["label"] for x in splits["train"]]

    val_texts = [x["text"] for x in splits["validation"]]
    val_labels = [x["label"] for x in splits["validation"]]

    logger.info("Tokenizing train and validation splits...")
    train_dataset = prepare_tensor_dataset(train_texts, train_labels, tokenizer, max_length=36)
    val_dataset = prepare_tensor_dataset(val_texts, val_labels, tokenizer, max_length=36)

    all_seed_results = []
    for seed in seeds:
        seed_res = train_single_seed(
            seed=seed,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=checkpoints_dir,
        )
        all_seed_results.append(seed_res)

    # Compute aggregate validation metrics across seeds (mean +- std)
    f1_list = [r["best_val_metrics"]["oos_f1"] for r in all_seed_results]
    recall_list = [r["best_val_metrics"]["oos_recall"] for r in all_seed_results]
    prec_list = [r["best_val_metrics"]["oos_precision"] for r in all_seed_results]
    acc_list = [r["best_val_metrics"]["in_scope_acceptance"] for r in all_seed_results]
    th_list = [r["best_val_threshold"] for r in all_seed_results]

    summary = {
        "seeds": list(seeds),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "aggregate_validation_metrics": {
            "val_oos_f1_mean": round(float(np.mean(f1_list)), 4),
            "val_oos_f1_std": round(float(np.std(f1_list)), 4),
            "val_oos_recall_mean": round(float(np.mean(recall_list)), 4),
            "val_oos_recall_std": round(float(np.std(recall_list)), 4),
            "val_oos_precision_mean": round(float(np.mean(prec_list)), 4),
            "val_oos_precision_std": round(float(np.std(prec_list)), 4),
            "val_in_scope_acceptance_mean": round(float(np.mean(acc_list)), 4),
            "val_in_scope_acceptance_std": round(float(np.std(acc_list)), 4),
            "val_threshold_mean": round(float(np.mean(th_list)), 4),
            "val_threshold_std": round(float(np.std(th_list)), 4),
        },
        "seed_runs": all_seed_results,
    }

    curves_file = results_dir / "training_curves.json"
    with open(curves_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved complete multi-seed training curves to {curves_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Learned Scope Gate across 3 seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    run_training_multi_seeds(
        seeds=args.seeds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
