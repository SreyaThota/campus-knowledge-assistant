# MLflow Experiment Tracking Summary: Learned Scope Gate

> **NOTICE: VALIDATION SPLIT TRACKING ONLY**
>
> All tracked runs, metrics, and thresholds logged below reflect tuning exclusively on the **CLINC150 validation split** ($N = 3,100$: $3,000$ in-scope, $100$ out-of-scope).
> **No test split data or campus evaluation data was used.**

---

## Experiment Metadata

- **Experiment Name:** `learned-scope-gate`
- **Parent Run ID:** `4bd4537bf8404ad7b88ff68b76faff2e`
- **Model Backbone:** `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters)
- **Classifier Head:** Mean Pooling → Dropout(0.2) → Linear(384 → 2)
- **Loss Function:** Class-Weighted CrossEntropyLoss (inverse frequency: $w_\text{oos} \approx 31.1$, $w_\text{in} \approx 0.51$)
- **Optimization:** AdamW (lr = 3e-5, weight decay = 0.01, linear warmup ratio = 0.1, max grad norm = 1.0)
- **Hardware Device:** CPU execution

---

## Logged Runs by Seed

| Run Name | Seed | MLflow Run ID | Best Epoch | Tuned Threshold ($\tau$) | Val OOS $F_1$ | Val OOS Recall | Val OOS Precision | Val In-Scope Acc | Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `learned_gate_seed_42` | 42 | `3ef0e261...` | 2 | 0.3858 | 0.7429 | 0.6500 | 0.8667 | 0.9967 | 2598.4 |
| `learned_gate_seed_1337` | 1337 | `2f493f02...` | 2 | 0.3072 | 0.7416 | 0.6600 | 0.8462 | 0.9960 | 1543.6 |
| `learned_gate_seed_2026` | 2026 | `8ffb84c6...` | 2 | 0.3219 | 0.7222 | 0.6500 | 0.8125 | 0.9950 | 1585.7 |

---

## Multi-Seed Validation Aggregate

- **Validation OOS $F_1$:** 0.7356 ± 0.0094
- **Validation OOS Recall:** 0.6533 ± 0.0047
- **Validation OOS Precision:** 0.8418 ± 0.0223
- **Validation In-Scope Acceptance:** 0.9959 ± 0.0007
- **Optimal Decision Threshold ($\tau$):** 0.3383 ± 0.0341

---

## How to Inspect Local MLflow Dashboard

To view the interactive MLflow UI with complete parameter diffs, epoch learning curves, and artifacts:
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```
