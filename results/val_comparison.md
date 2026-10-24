# Scope Gate Validation-Only Baseline Comparison

> **NOTICE: VALIDATION SPLIT ONLY**
> 
> All metrics and decision thresholds reported in this document were evaluated strictly on the **CLINC150 validation split** ($N = 3{,}100$: $3{,}000$ in-scope, $100$ out-of-scope).
> 
> - **Zero Test Split Access:** The CLINC150 test split ($N = 5{,}500$) and campus evaluation set ($N = 110$) have **not** been touched.
> - **Threshold Tuning:** All decision thresholds ($\tau$) were chosen strictly on the validation split using the maximum out-of-scope (OOS) $F_1$ criterion.
> - **Preliminary Status:** These numbers are strictly for architecture selection and validation check. They must **not** be cited in the main README as final model evaluation numbers.

---

## Validation Benchmark Table

> **Definition:** *False Rejections (FP)*: In-scope student queries that the gate wrongly refused as out-of-scope.

| Method | Decision Threshold ($\tau_{\text{val}}$) | OOS $F_1$ | OOS Recall | OOS Precision | In-Scope Acceptance | False Rejections (FP / 3,000) | FN (OOS / 100) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.5725 | 0.4946 | 0.4600 | 0.5349 | 0.9867 | 40 | 54 |
| **Centroid Cosine Similarity** | 0.8918 | 0.3544 | 0.5900 | 0.2532 | 0.9420 | 174 | 41 |
| **Exemplar Cosine Similarity (Nearest Neighbor)** | 0.5504 | 0.6224 | 0.6100 | 0.6354 | 0.9883 | 35 | 39 |
| **Frozen Embedding + Logistic Regression** | 0.8819 | 0.4878 | 0.4000 | 0.6250 | 0.9920 | 24 | 60 |
| **Learned Scope Gate (Seed 42)** | 0.3858 | 0.7429 | 0.6500 | 0.8667 | 0.9967 | 10 | 35 |
| **Learned Scope Gate (Seed 1337)** | 0.3072 | 0.7416 | 0.6600 | 0.8462 | 0.9960 | 12 | 34 |
| **Learned Scope Gate (Seed 2026)** | 0.3219 | 0.7222 | 0.6500 | 0.8125 | 0.9950 | 15 | 35 |
| **Learned Scope Gate (Mean ± Std across 3 seeds)** | **0.3383 ± 0.0341** | **0.7356 ± 0.0094** | **0.6533 ± 0.0047** | **0.8418 ± 0.0223** | **0.9959 ± 0.0007** | **12.3 ± 2.1** | **34.7 ± 0.5** |

---

## Decision Surface Analysis: Threshold Sweep ($\tau \in [0.20, 0.50]$)

Measured on the CLINC validation split ($N = 3{,}100$) across 3 training seeds (42, 1337, 2026):

| Threshold ($\tau$) | OOS Recall (Mean ± Std) | OOS Precision (Mean ± Std) | OOS $F_1$ (Mean ± Std) | In-Scope Acceptance | False Rejections (FP / 3,000) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.20** | 0.6633 ± 0.0047 | 0.7809 ± 0.0180 | 0.7172 ± 0.0057 | 0.9938 ± 0.0007 | 18.7 ± 2.1 |
| **0.25** | 0.6600 ± 0.0000 | 0.7957 ± 0.0204 | 0.7214 ± 0.0085 | 0.9943 ± 0.0007 | 17.0 ± 2.2 |
| **0.30** | 0.6567 ± 0.0047 | 0.8110 ± 0.0180 | 0.7257 ± 0.0095 | 0.9949 ± 0.0006 | 15.3 ± 1.7 |
| **0.35** | 0.6467 ± 0.0094 | 0.8257 ± 0.0131 | 0.7252 ± 0.0076 | 0.9954 ± 0.0004 | 13.7 ± 1.2 |
| **0.40** | 0.6300 ± 0.0141 | 0.8403 ± 0.0270 | 0.7201 ± 0.0191 | 0.9960 ± 0.0007 | 12.0 ± 2.2 |
| **0.45** | 0.6100 ± 0.0163 | 0.8438 ± 0.0335 | 0.7081 ± 0.0225 | 0.9962 ± 0.0009 | 11.3 ± 2.6 |
| **0.50** | 0.5967 ± 0.0094 | 0.8449 ± 0.0268 | 0.6993 ± 0.0145 | 0.9963 ± 0.0007 | 11.0 ± 2.2 |

### Empirical Finding on Decision Surface
- **No Steep Cliff:** The empirical sweep refutes the hypothesis of a steep decision cliff where a $\pm 0.05$ threshold adjustment causes wild metric swings.
- **Plateau Behavior:** Between $\tau = 0.25$ and $\tau = 0.40$, $F_1$ forms a smooth plateau ($0.720\text{--}0.726$), recall moves by only ~3 percentage points ($66.0\% \to 63.0\%$), and in-scope acceptance remains exceptionally high and stable ($99.43\% \to 99.60\%$).
- **Trade-Off:** Moving from $\tau = 0.20$ to $0.50$ (a 0.30 span) trades 6.7 points of recall ($66.3\% \to 59.7\%$) for 6.4 points of precision ($78.1\% \to 84.5\%$) and reduces false rejections from 18.7 down to 11.0 out of 3,000 queries.

---

## Statistical Uncertainty: Bootstrap 95% Confidence Intervals

Computed via non-parametric bootstrap ($B = 1{,}000$ resamples) on the validation split ($N = 3{,}100$):

| Method / Seed | Decision Threshold ($\tau$) | OOS $F_1$ [95% CI] | OOS Recall [95% CI] | OOS Precision [95% CI] | In-Scope Acceptance [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exemplar Cosine Baseline** | 0.5504 | 0.6154 [0.5301, 0.6909] | 0.6000 [0.5000, 0.6939] | 0.6316 [0.5326, 0.7284] | 0.9883 [0.9843, 0.9923] |
| **Learned Scope Gate (Seed 42)** | 0.3858 | 0.7429 [0.6667, 0.8114] | 0.6500 [0.5607, 0.7411] | 0.8667 [0.7805, 0.9342] | 0.9967 [0.9943, 0.9983] |
| **Learned Scope Gate (Seed 1337)** | 0.3072 | 0.7416 [0.6667, 0.8093] | 0.6600 [0.5700, 0.7525] | 0.8462 [0.7595, 0.9243] | 0.9960 [0.9934, 0.9980] |
| **Learned Scope Gate (Seed 2026)** | 0.3219 | 0.7222 [0.6437, 0.7919] | 0.6500 [0.5565, 0.7383] | 0.8125 [0.7179, 0.8939] | 0.9950 [0.9923, 0.9973] |

### Assessment of the 11-Point $F_1$ Gap
1. **Point Estimate Advantage:** The point-estimate mean $F_1$ of the Learned Gate ($0.7356$) leads the Exemplar Cosine baseline ($0.6224 / 0.6154$) by $\approx +11.3$ points.
2. **Confidence Interval Overlap:** Due to the small validation sample ($N_{\text{oos}} = 100$), the 95% bootstrap confidence intervals span $\approx 15\text{--}16$ percentage points. The upper bound of the Exemplar Cosine CI ($0.6909$) overlaps with the lower bounds of the Learned Gate models ($0.6437$ for seed 2026, $0.6667$ for seeds 42 and 1337).
3. **Statistical Significance Conclusion:** While the Learned Gate consistently achieves higher point estimates across all seeds and cuts false rejections by more than half ($12.3$ vs $35$), the 11-point $F_1$ gap cannot be claimed as statistically non-overlapping on validation alone. Firm statistical separation will be verified on the held-out test split ($N_{\text{oos}} = 1{,}000$).

---

## Validation Sample Size & Tuning Limitations

1. **Small Out-of-Scope Sample ($N = 100$):**
   - The CLINC validation split contains exactly 100 OOS queries. Consequently, **each correctly or incorrectly classified query moves recall by exactly 1.0 percentage point**.
2. **Dual-Use of the Validation Split:**
   - The validation split was used for both **decision threshold selection** ($\arg\max_{\tau} F_1$) and **comparative architecture selection** against baselines.
   - This creates an inherent risk of optimistic selection bias; final generalization must be validated on the untouched test split.
