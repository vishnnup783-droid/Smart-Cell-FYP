"""
SmartCell project - Baseline SOH / RUL classification on the NASA PCoE
Li-ion Battery Data Set (B0005, B0006, B0007, B0018).

WHY THIS EXISTS
----------------
There is no standard public dataset for 12V lead-acid batteries in real
two-wheeler use, so this uses the NASA Li-ion dataset (the most widely used
public battery-aging dataset) purely to build and validate a SOH/RUL
*classification methodology* -- feature extraction from a discharge curve,
class-label construction, battery-wise train/test splitting, model training
and evaluation. The feature types used here (discharge-curve shape,
time-to-cutoff, temperature behaviour) map onto signals your own ESP32
pipeline already measures, so the exercise transfers even though the cell
chemistry differs. The deployed on-device algorithm should remain the
simpler, explainable classical SOH/RUL estimator recommended in the Revised
Directions document -- the Random Forest feature-importance output this
produces is evidence for why that pivot is reasonable.

CODE LAYOUT
-----------
smartcell_ml/config.py    All settings: paths, thresholds, feature list, model hyperparams.
smartcell_ml/data.py      Load .mat files, extract per-cycle features, build SOH/RUL labels.
smartcell_ml/evaluate.py  Leave-one-battery-out training + metrics for one target column.
smartcell_ml/plots.py     Confusion matrices, capacity-fade curve, feature-importance chart.
smartcell_ml/pipeline.py  Wires the above together and writes outputs/.
This file is just the entry point - run it directly:

    python3 soh_rul_pipeline.py
"""

from smartcell_ml.pipeline import main

if __name__ == "__main__":
    main()
