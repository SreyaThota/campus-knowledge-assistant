# Scope Gate Validation-Only Baseline Comparison

> **NOTICE: VALIDATION SPLIT ONLY**
> 
> All metrics and decision thresholds reported in this document were evaluated strictly on the **CLINC150 validation split** ($N = 3{,}100$: $3{,}000$ in-scope, $100$ out-of-scope).
> 
> - **Zero Test Split Access:** The CLINC150 test split ($N = 5{,}500$) and campus evaluation set ($N = 110$) have **not** been touched.
> - **Threshold Tuning:** All decision thresholds ($\tau$) were chosen strictly on the validation split using the maximum out-of-scope (OOS) $F_1$ criterion.
> - **Preliminary Status:** These numbers are strictly for architecture selection and validation check. They must **not** be cited in the main README or reported as final model evaluation numbers.

---

## Validation Benchmark Table

| Method | Decision Threshold ($\tau_{\text{val}}$) | OOS $F_1$ | OOS Recall | OOS Precision | In-Scope Acceptance | TP (OOS) | FP (In-Scope) | FN (OOS) | TN (In-Scope) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.5725 | 0.4946 | 0.4600 | 0.5349 | 0.9867 | 46 | 40 | 54 | 2960 |
| **Centroid Cosine Similarity** | 0.8918 | 0.3544 | 0.5900 | 0.2532 | 0.9420 | 59 | 174 | 41 | 2826 |
| **Exemplar Cosine Similarity (Nearest Neighbor)** | 0.5504 | 0.6224 | 0.6100 | 0.6354 | 0.9883 | 61 | 35 | 39 | 2965 |
| **Frozen Embedding + Logistic Regression** | 0.8819 | 0.4878 | 0.4000 | 0.6250 | 0.9920 | 40 | 24 | 60 | 2976 |
| **Learned Scope Gate (Seed 42)** | 0.3858 | 0.7429 | 0.6500 | 0.8667 | 0.9967 | 65 | 10 | 35 | 2990 |
| **Learned Scope Gate (Seed 1337)** | 0.3072 | 0.7416 | 0.6600 | 0.8462 | 0.9960 | 66 | 12 | 34 | 2988 |
| **Learned Scope Gate (Seed 2026)** | 0.3219 | 0.7222 | 0.6500 | 0.8125 | 0.9950 | 65 | 15 | 35 | 2985 |
| **Learned Scope Gate (Mean ± Std across 3 seeds)** | **0.3383 ± 0.0341** | **0.7356 ± 0.0094** | **0.6533 ± 0.0047** | **0.8418 ± 0.0223** | **0.9959 ± 0.0007** | **65.3 ± 0.5** | **12.3 ± 2.1** | **34.7 ± 0.5** | **2987.7 ± 2.1** |

---

## Experimental Protocol & Parameters

1. **Dataset Split Distribution (Validation):**
   - Total queries: $3{,}100$
   - Ground truth in-scope queries: $3{,}000$ ($96.77\%$)
   - Ground truth out-of-scope (OOS) queries: $100$ ($3.23\%$)
   - Class imbalance: ~30:1 in validation (and ~60:1 in training: 15,000 in-scope to 245 OOS queries).

2. **Threshold Tuning Criterion:**
   - Threshold sweep: 400 candidate thresholds uniformly spaced in $[0.01, 0.99]$.
   - Criterion: $\arg\max_{\tau} F_1(\text{OOS})$.
   - Decision rule: A query is rejected as out-of-scope if $P(\text{OOS}) \ge \tau$, otherwise accepted as in-scope.

3. **Key Observations:**
   - **Baseline Comparison:** The Learned Scope Gate outperforms the nearest-neighbor baseline (Exemplar Cosine) by $+11.32$ $F_1$ points on validation ($0.7356$ vs $0.6224$).
   - **Precision & User Experience:** Precision improves from $63.54\%$ to $84.18\%$ ($+20.64$ percentage points), reducing false rejections from 35 down to 10–15 queries out of 3,000 ($99.59\%$ in-scope acceptance).
   - **Ablation vs Frozen Embeddings:** End-to-end transformer fine-tuning beats frozen embeddings + logistic regression by $+24.78$ $F_1$ points ($0.7356$ vs $0.4878$), confirming that adaptation of intermediate encoder representations is necessary under severe class imbalance.
   - **Seed Stability:** Variance across seeds is low ($\sigma_{F_1} = 0.0094$), showing consistent optimization dynamics.
