# Physics-Grounded Latent Inversion with Selective State-Space Learning for In-Line PCCP Inspection

![methodology](/assets/PISSM_v2.png)

In-line eddy-current testing (ECT) of buried prestressed concrete cylinder pipe (PCCP) produces long dual-frequency impedance records whose statistics shift between inspection campaigns, and learned diagnostic models that perform well on one campaign often fail silently on the next. This article presents a two-stage interpretation framework that treats campaign drift as physics to be inverted rather than noise to be normalized. A model-based latent inversion fits each chainage-indexed record against a closed-form Dodd–Deeds forward operator for the finite-thickness steel cylinder, jointly estimating an effective conductivity–permeability state, a per-sequence channel calibration, and a physics-normalized anomaly stream; boundary conditions are satisfied analytically, and no material constants are assumed. A bidirectional selective state-space network then performs joint boundary localization and severity classification on the inverted features, with abstention driven by dual-detector agreement.

本文提出一種兩階段架構,將埋地預應力鋼筒混凝土管(PCCP)渦流檢測中的跨批次漂移視為可反演的物理量:以Dodd–Deeds解析正演算子進行模型式潛變量反演(有效電導率–磁導率狀態、逐序列校準、物理歸一化異常流),再由選擇性狀態空間網路完成接頭定位.

## Contributions:
- First physics-informed, velocity-normalized multi-channel selective-SSM for long-range NDT.
- Learned low-rank common-mode baseline as the drift-robust, label-free substitute for physics.
- Spatial×spectral×detector fusion justified by measured complementarity/contrast.
- Genuine linear-complexity long-range inference where prior work used quadratic Transformers or fabricated state-space models.

## Experiment Plan — SST-SSM for ECT Pipe Inspection

## Claims

| ID | Claim | Hypothesis | Primary metric | Pass bar |
|----|-------|------------|----------------|----------|
| **C1** | PISSM jointly segments + classifies better than the existing baselines end-to-end | coupling + long-range SSM reduces error propagation | Propagated Precision (PP) | PP ≥ best baseline + 0.03 |
| **C2** | Low-rank common-mode removal improves **drift/OOD** robustness | drift is common-mode; removing it yields drift-invariant signal | ΔF1 = F1_ID − F1_OOD (lower better) | Δ smaller than "w/o common-mode" by ≥0.03 |
| **C3** | Dual-frequency fusion beats single-frequency | 32/100 Hz are complementary (corr 0.4–0.77) | weighted F1 | F1w(both) > F1w(best single) |
| **C4** | Selective SSM gives linear-complexity long-range modeling vs Transformer | O(L) vs O(L²) | wall-clock + peak mem vs seq len | linear scaling to ≥100k; < Transformer mem |
| **C5** | Cross-detector contrast \|D1−D2\| adds classification signal | measured discriminative | per-class F1 (rare classes) | F1(with contrast) > F1(without) on minority |
| **C6** | Velocity normalization improves boundary localization | removes 18±16mm spatial jitter | Boundary MAD (m) | MAD(resampled) < MAD(raw index) |

## Ablation matrix (each row = one trained model, same data/splits/seeds)
| Config | Velocity norm | Common-mode removal | Dual-freq | \|D1−D2\| contrast | SSM backbone | Coupled heads |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full (SST-SSM)** | ✓ | ✓ | ✓ | ✓ | bi-SSM | ✓ |
| w/o velocity norm | ✗ | ✓ | ✓ | ✓ | bi-SSM | ✓ |
| w/o common-mode | ✓ | ✗ | ✓ | ✓ | bi-SSM | ✓ |
| single-freq (32) | ✓ | ✓ | 32-only | ✓ | bi-SSM | ✓ |
| w/o contrast | ✓ | ✓ | ✓ | ✗ | bi-SSM | ✓ |
| SSM→GRU | ✓ | ✓ | ✓ | ✓ | biGRU | ✓ |
| SSM→Transformer | ✓ | ✓ | ✓ | ✓ | attn | ✓ |
| decoupled heads | ✓ | ✓ | ✓ | ✓ | bi-SSM | ✗ |

Each cell reports MAD(m), Seg-Acc@tol, F1w, PP. Maps directly to C1–C6.

## Baselines (existing real code, same protocol)
CNN-LSTM (`multiclass_final.py`), TCN (`segment_detector_TCN.py`), MTF-CNN
(`mtf_binary_classifier_fixed.py`), CPD (`segment_detector_cpd.py`), GaussianHMM (`SSSM_v2.py`).

## Data splits
- **ID:** train/val/test split *within* a campaign pool {102, 88003, ...}.
- **OOD (cross-campaign):** train on a set of contracts, test on held-out contracts
  (files differ in class balance: 102=27% fault, 88003=1.8%, 86002=2%, 815002=0%).
- Report ID and OOD for every model. Seeds: {42, 123, 2024} → mean ± std.

## Metrics (all in `ssm_ndt/metrics.py`, computed from committed predictions)
- **Boundary MAD (m):** greedy-match predicted boundary peaks to true, mean |Δ| × grid spacing.
- **Seg-Acc@tol:** fraction of true boundaries matched within tolerance τ (default 0.5 m).
- **F1 / weighted F1:** per-class and class-frequency-weighted.
- **Propagated Precision:** class-correct only where boundary localized within τ.
- **Throughput/mem (C4):** samples/s and peak memory vs sequence length.

## Imbalance handling
Existing GAN augmentation (`synthetic_GAN.py`) + class-weighted CE + minority-focused F1 reporting.

## Definition of done
A results table + ablation table generated entirely by `ssm_ndt/train.py` →
`results/*.json`, plus figures, with seeds and OOD columns. Then and only then, claims are written.

## Run order
1. `ssm_ndt/data.py` smoke (load + resample + decompose) ✅ first.
2. Train Full on one campaign (overfit sanity) → metrics sane.
3. Ablation rows on the campaign pool (ID).
4. Cross-campaign OOD for Full + key ablations.
5. C4 complexity benchmark.
6. Baseline re-evaluation under same metrics.
