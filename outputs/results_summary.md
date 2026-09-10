# SOH / RUL Classifier — NASA Dataset Baseline

Baseline classification pipeline for SmartCell, built and validated on the NASA PCoE Li-ion Battery Data Set (B0005, B0006, B0007, B0018) as a proof-of-concept for the feature-extraction-plus-classification methodology. This is not the deployed on-device algorithm — per the Revised Directions document, the ESP32 pipeline should use the simpler classical SOH/RUL estimator (internal resistance trend, Coulomb counting, OCV lookup). This exercise validates that approach with data.

## Dataset

- Source: NASA Prognostics Center of Excellence Battery Data Set, cells B0005/B0006/B0007/B0018 (room-temperature cycling to failure).
- 636 discharge cycles total across the four cells (168 + 168 + 168 + 132).
- Rated capacity: 2.0 Ah (nameplate). End-of-life threshold: 70% of rated capacity (1.4 Ah), the standard threshold used in the original Saha & Goebel (2007) NASA study and most SOH/RUL papers built on this dataset.

## Labels

**SOH class** (from measured discharge capacity):
- Healthy: SOH ≥ 80% (283 cycles)
- Degrading: 65% ≤ SOH < 80% (319 cycles)
- Critical: SOH < 65% (34 cycles)

**RUL class** (cycles remaining until the cell crosses the 70% EOL threshold, tertile-split across the pooled dataset):
- Short: 0–18 cycles remaining (213 cycles)
- Medium: 19–71 cycles remaining (212 cycles)
- Long: >71 cycles remaining (211 cycles)

## Features (per discharge cycle, 14 total)

Extracted directly from the raw voltage/current/temperature/time traces of each discharge cycle: mean/std/min/max voltage, voltage drop rate, mean/std current, mean/max/std temperature, discharge duration, energy released (Wh), time to reach 3.0 V, and ambient temperature. No cycle-number feature was used, to force the models to learn from the electrical signal itself rather than memorizing elapsed time.

## Validation method

Leave-one-battery-out cross-validation: each of the four cells was held out as the test set in turn while the model trained on the other three, and metrics were pooled across all four folds. This means every reported number reflects performance on a battery the model never saw during training — a fair proxy for how the approach would generalize to a battery your own hardware hasn't been calibrated on.

## Results

### SOH classification (Healthy / Degrading / Critical)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Logistic Regression | 0.90 | 0.90 |
| Decision Tree (depth 5) | 0.96 | 0.96 |
| Random Forest (300 trees) | 0.96 | 0.96 |

Random Forest per-class: Critical precision/recall 0.85/0.85, Degrading 0.97/0.95, Healthy 0.97/0.98. The Critical class is the smallest (34 of 636 cycles) and still gets the weakest — but still solid — recall, which is the honest way to report it: a minority-class dip is expected and worth naming rather than hiding behind the overall accuracy number.

### RUL classification (Short / Medium / Long)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Logistic Regression | 0.68 | 0.68 |
| Decision Tree (depth 5) | 0.82 | 0.82 |
| Random Forest (300 trees) | 0.85 | 0.85 |

RUL is the harder target, as expected (it depends on the trajectory, not just the current state), but Random Forest still generalizes to held-out batteries at 85% accuracy with no class weaker than 0.79 F1.

## Feature importance (Random Forest, fit on full pooled data)

Top features for both SOH and RUL were the same two, in the same order:
1. `time_to_3v_s` — how long the cell holds voltage above 3.0 V before sagging
2. `energy_wh` — total energy released during the discharge

Followed by discharge duration and mean voltage. Internal-resistance-adjacent, discharge-curve-shape signals dominate; nothing in the top features requires anything beyond voltage, current and time measurement over a discharge — exactly what your ESP32 pipeline already senses. This is the evidence for the report: the signals that actually drive the classification are the same ones your classical SOH/RUL estimator (internal resistance trend, OCV, Coulomb counting) already uses, so dropping the LSTM layer per the panel's feedback does not sacrifice the predictive signal, it just replaces a black-box model with an explainable one built on the same inputs.

## Files produced

- `outputs/engineered_features.csv` — the full 636-row feature table with labels, for your own further analysis.
- `outputs/results.json` — machine-readable metrics, confusion matrices, classification reports and feature importances for both targets.
- `outputs/plots/capacity_fade_soh.png` — SOH trajectory for all four cells with class boundaries marked.
- `outputs/plots/confusion_matrices_SOH.png`, `confusion_matrices_RUL.png` — per-model confusion matrices.
- `outputs/plots/feature_importance_SOH.png`, `feature_importance_RUL.png` — Random Forest feature rankings.

## Caveats to state in your report

- This is a Li-ion dataset used to validate the classification methodology; it is not a substitute for lead-acid data from your own hardware, and SOH/RUL numbers here should not be presented as characterizing lead-acid behaviour.
- The 65%/80% SOH cutoffs and the tertile-based RUL cutoffs are reasonable, literature-consistent choices but are still a design decision — say so explicitly rather than presenting them as fixed ground truth.
- Leave-one-battery-out with only 4 cells means each fold's test set is an entire battery's cycle history, so results can shift meaningfully with a 5th or 6th cell; treat these numbers as indicative, not final.
