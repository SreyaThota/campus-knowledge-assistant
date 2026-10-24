# Final Held-Out Evaluation Summary

> **Protocol Integrity:**
> - Evaluated strictly on frozen held-out datasets:
>   - **CLINC150 Test Split:** $N = 5{,}500$ ($4{,}500$ in-scope, $1{,}000$ out-of-scope).
>   - **Campus Evaluation Set:** $N = 140$ ($85$ in-scope, $55$ out-of-scope), frozen under SHA-256 checksum `69fb89da7af6e2e628937c160824305d46487bfee519bcd29b6a2802ec31c650`.
> - **Zero Test Leakage:** All decision thresholds ($\tau$) were tuned strictly on the validation split and frozen without modification.
> - **Bootstrap Resampling:** 95% confidence intervals computed via $B = 1{,}000$ empirical percentile bootstrap resamples.

---

## 1. CLINC150 Test Split Benchmark ($N = 5{,}500$)

> **Definition:** *False Rejections (FP)*: In-scope test queries wrongly refused by the gate as out-of-scope.

| Method | Frozen Threshold ($\tau$) | OOS $F_1$ [95% CI] | OOS Recall [95% CI] | OOS Precision [95% CI] | In-Scope Acceptance [95% CI] | False Rejections (FP / 4,500) | FN (OOS / 1,000) | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.5725 | 0.4129 [0.3793, 0.4458] | 0.2760 [0.2483, 0.3043] | 0.8190 [0.7768, 0.8584] | 0.9864 [0.9830, 0.9898] | 61 | 724 | 0.0309 |
| **Centroid Cosine Similarity** | 0.8918 | 0.4219 [0.3922, 0.4527] | 0.3320 [0.3052, 0.3604] | 0.5784 [0.5401, 0.6184] | 0.9462 [0.9399, 0.9526] | 242 | 668 | 0.5488 |
| **Exemplar Cosine Similarity (Nearest Neighbor)** | 0.5504 | **0.6230 [0.5940, 0.6514]** | **0.4710 [0.4410, 0.5020]** | 0.9199 [0.8959, 0.9439] | 0.9909 [0.9881, 0.9936] | 41 | 529 | 0.1458 |
| **Frozen Embedding + Logistic Regression** | 0.8598 | 0.4530 [0.4206, 0.4853] | 0.3060 [0.2782, 0.3344] | 0.8718 [0.8352, 0.9066] | 0.9900 [0.9869, 0.9929] | 45 | 694 | 0.0562 |
| **Learned Scope Gate (Seed 42)** | 0.3858 | 0.5402 [0.5085, 0.5718] | 0.3760 [0.3461, 0.4065] | 0.9592 [0.9393, 0.9757] | 0.9964 [0.9947, 0.9980] | 16 | 624 | 0.1100 |
| **Learned Scope Gate (Seed 1337)** | 0.3072 | 0.5300 [0.4968, 0.5619] | 0.3710 [0.3402, 0.4014] | 0.9275 [0.9015, 0.9516] | 0.9936 [0.9912, 0.9956] | 29 | 629 | 0.1125 |
| **Learned Scope Gate (Seed 2026)** | 0.3219 | 0.5780 [0.5474, 0.6088] | 0.4170 [0.3870, 0.4484] | 0.9413 [0.9202, 0.9614] | 0.9942 [0.9921, 0.9962] | 26 | 583 | 0.1045 |
| **Learned Scope Gate (Mean ± Std)** | **—** | **0.5494 ± 0.0206** | **0.3880 ± 0.0206** | **0.9427 ± 0.0130** | **0.9947 ± 0.0012** | **23.7 ± 5.7** | **612.0 ± 20.6** | **0.1090 ± 0.0034** |

### Key Test Findings
1. **Baseline Beat Learned Gate on Test $F_1$:** Exemplar Cosine Nearest Neighbor outperformed the Learned Gate ($0.6230$ vs $0.5494$, a +7.36 point gap).
2. **Generalization Gap:** On validation, the Learned Gate achieved $F_1 = 0.7356$; on held-out test, $F_1$ dropped to $0.5494$ (-18.62 points). The drop was driven entirely by a collapse in OOS recall ($65.3\% \to 38.8\%$), whereas precision remained exceptionally strong ($94.3\%$, only 24 false rejections out of 4,500 in-scope queries).
3. **Exemplar Stability:** Exemplar Cosine showed near-zero transfer gap between splits ($F_1 = 0.6224$ on val vs $0.6230$ on test), proving that non-parametric nearest neighbor retrieval is less prone to validation threshold overfitting under class imbalance.

---

## 2. Frozen Campus Evaluation Set Benchmark ($N = 140$)

### Overall Campus Benchmark ($N = 140$: 85 In-Scope, 55 Out-of-Scope)

| Method | Threshold | OOS $F_1$ [95% CI] | OOS Recall [95% CI] | OOS Precision [95% CI] | In-Scope Acceptance [95% CI] | False Rejections (FP / 85) | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.5725 | 0.3820 [0.2500, 0.5000] | 0.3091 [0.1875, 0.4311] | 0.5000 [0.3413, 0.6774] | 0.8000 [0.7143, 0.8877] | 17 | 0.2314 |
| **Centroid Cosine Similarity** | 0.8918 | 0.3540 [0.2407, 0.4629] | 0.3636 [0.2391, 0.4920] | 0.3448 [0.2308, 0.4590] | 0.5529 [0.4444, 0.6506] | 38 | 0.4685 |
| **Exemplar Cosine Similarity** | 0.5504 | 0.4462 [0.3306, 0.5442] | 0.5273 [0.3888, 0.6471] | 0.3867 [0.2812, 0.4940] | 0.4588 [0.3552, 0.5658] | 46 | 0.2241 |
| **Frozen Embedding + Logistic Regression** | 0.8598 | 0.2759 [0.1579, 0.3956] | 0.2182 [0.1186, 0.3274] | 0.3750 [0.2222, 0.5600] | 0.7647 [0.6738, 0.8534] | 20 | 0.3143 |
| **Learned Scope Gate (Seed 42)** | 0.3858 | 0.4643 [0.3434, 0.5737] | 0.4727 [0.3386, 0.6000] | 0.4561 [0.3333, 0.5910] | 0.6353 [0.5352, 0.7363] | 31 | 0.3622 |
| **Learned Scope Gate (Seed 1337)** | 0.3072 | 0.4779 [0.3542, 0.5882] | 0.4909 [0.3509, 0.6182] | 0.4655 [0.3454, 0.5965] | 0.6353 [0.5333, 0.7403] | 31 | 0.3459 |
| **Learned Scope Gate (Seed 2026)** | 0.3219 | 0.4915 [0.3784, 0.5968] | 0.5273 [0.3953, 0.6508] | 0.4603 [0.3448, 0.5834] | 0.6000 [0.4999, 0.7011] | 34 | 0.3379 |
| **Learned Scope Gate (Mean ± Std)** | **—** | **0.4779 ± 0.0111** | **0.4970 ± 0.0227** | **0.4606 ± 0.0038** | **0.6235 ± 0.0166** | **32.0 ± 1.4** | **0.3487 ± 0.0101** |

---

### Subset Breakdown: Author (`human` vs `ai_draft`)

#### Human Subset ($N = 30$: 15 In-Scope, 15 Out-of-Scope)
| Method | OOS $F_1$ [95% CI] | OOS Recall [95% CI] | OOS Precision [95% CI] | In-Scope Acceptance [95% CI] | False Rejections (FP / 15) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.1176 [0.0000, 0.3478] | 0.0667 [0.0000, 0.2222] | 0.5000 [0.0000, 1.0000] | 0.9333 [0.7778, 1.0000] | 1 |
| **Centroid Cosine Similarity** | 0.3846 [0.1053, 0.6000] | 0.3333 [0.0909, 0.5791] | 0.4545 [0.1663, 0.7504] | 0.6000 [0.3333, 0.8462] | 6 |
| **Exemplar Cosine Similarity** | 0.5556 [0.3333, 0.7393] | 0.6667 [0.3889, 0.9000] | 0.4762 [0.2632, 0.6818] | 0.2667 [0.0587, 0.5000] | 11 |
| **Frozen Embedding + Logistic Regression** | 0.2727 [0.0000, 0.5000] | 0.2000 [0.0000, 0.4212] | 0.4286 [0.0000, 0.8000] | 0.7333 [0.4993, 0.9333] | 4 |
| **Learned Scope Gate (Seed 42)** | 0.5385 [0.2665, 0.7500] | 0.4667 [0.2000, 0.7273] | 0.6364 [0.3333, 0.9002] | 0.7333 [0.5000, 0.9286] | 4 |
| **Learned Scope Gate (Seed 1337)** | 0.5385 [0.2727, 0.7500] | 0.4667 [0.2143, 0.7273] | 0.6364 [0.3333, 0.9091] | 0.7333 [0.5000, 0.9287] | 4 |
| **Learned Scope Gate (Seed 2026)** | **0.5714 [0.3333, 0.7742]** | 0.5333 [0.2854, 0.8000] | **0.6154 [0.3333, 0.8750]** | 0.6667 [0.4167, 0.8825] | 5 |
| **Learned Scope Gate (Mean ± Std)** | **0.5495 ± 0.0155** | **0.4889 ± 0.0314** | **0.6294 ± 0.0099** | **0.7111 ± 0.0314** | **4.3 ± 0.5** |

#### AI Draft Subset ($N = 110$: 70 In-Scope, 40 Out-of-Scope)
| Method | OOS $F_1$ [95% CI] | OOS Recall [95% CI] | OOS Precision [95% CI] | In-Scope Acceptance [95% CI] | False Rejections (FP / 70) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.4444 [0.2903, 0.5854] | 0.4000 [0.2439, 0.5625] | 0.5000 [0.3333, 0.6774] | 0.7714 [0.6709, 0.8630] | 16 |
| **Centroid Cosine Similarity** | 0.3448 [0.2247, 0.4707] | 0.3750 [0.2368, 0.5366] | 0.3191 [0.1957, 0.4546] | 0.5429 [0.4225, 0.6618] | 32 |
| **Exemplar Cosine Similarity** | 0.4043 [0.2696, 0.5218] | 0.4750 [0.3233, 0.6316] | 0.3519 [0.2273, 0.4817] | 0.5000 [0.3750, 0.6191] | 35 |
| **Frozen Embedding + Logistic Regression** | 0.2769 [0.1389, 0.4118] | 0.2250 [0.1052, 0.3684] | 0.3600 [0.1786, 0.5556] | 0.7714 [0.6665, 0.8710] | 16 |
| **Learned Scope Gate (Seed 42)** | 0.4419 [0.3170, 0.5743] | 0.4750 [0.3333, 0.6305] | 0.4130 [0.2790, 0.5610] | 0.6143 [0.5000, 0.7314] | 27 |
| **Learned Scope Gate (Seed 1337)** | 0.4598 [0.3371, 0.5769] | 0.5000 [0.3513, 0.6445] | 0.4255 [0.2979, 0.5715] | 0.6143 [0.4930, 0.7344] | 27 |
| **Learned Scope Gate (Seed 2026)** | 0.4667 [0.3414, 0.5870] | 0.5250 [0.3750, 0.6744] | 0.4200 [0.2979, 0.5625] | 0.5857 [0.4697, 0.7027] | 29 |
| **Learned Scope Gate (Mean ± Std)** | **0.4561 ± 0.0105** | **0.5000 ± 0.0204** | **0.4195 ± 0.0051** | **0.6048 ± 0.0135** | **27.7 ± 0.9** |

---

### Subset Breakdown: Language & Difficulty

| Subset | Method | OOS $F_1$ | OOS Recall | OOS Precision | In-Scope Acceptance | False Rejects |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English ($N = 109$)** | Exemplar Cosine | 0.4396 | 0.4444 | 0.4348 | 0.5938 | 26 / 64 |
| | Learned Gate (Mean) | **0.4802** | **0.5037** | **0.4598** | **0.5833** | **26.7 / 64** |
| **German ($N = 31$)** | TF-IDF + LogReg | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 / 21 |
| | Exemplar Cosine | 0.4615 | **0.9000** | 0.3103 | 0.0476 (Catastrophic) | 20 / 21 |
| | Learned Gate (Seed 2026) | **0.5000** | 0.5000 | **0.5000** | **0.7619** | **5 / 21** |
| **Easy ($N = 89$)** | Exemplar Cosine | 0.3377 | 0.4815 | 0.2600 | 0.4032 | 37 / 62 |
| | Learned Gate (Mean) | **0.3852** | 0.4444 | **0.3400** | **0.6237** | **23.3 / 62** |
| **Hard ($N = 51$)** | Exemplar Cosine | 0.6038 | 0.5714 | 0.6400 | 0.6087 | 9 / 23 |
| | Learned Gate (Seed 2026) | **0.6296** | **0.6071** | **0.6538** | **0.6087** | **9 / 23** |

---

## 3. Critical Failures & Insights

1. **Catastrophic Out-of-Domain False Rejections:**
   - On the CLINC distribution, the Learned Gate achieved $99.5\%$ in-scope acceptance (only 24 false rejections out of 4,500).
   - On the campus distribution, in-scope acceptance dropped to **$62.35\%$** (32 out of 85 valid campus queries were rejected).
   - Exemplar Cosine suffered even worse: in-scope acceptance fell to **$45.88\%$** overall, and **$4.76\%$** on German (it rejected 20 out of 21 legitimate German campus questions).
2. **Nearest-Neighbor Baseline Won on Test:**
   - Fine-tuned transformer gate overfit its decision threshold on validation; on held-out test, nearest-neighbor exemplar retrieval achieved higher $F_1$ ($0.6230$ vs $0.5494$).
