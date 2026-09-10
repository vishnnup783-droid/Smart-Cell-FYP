# SmartCell — SOH/RUL Classifier Baseline (NASA Dataset)

Baseline classification pipeline built on the NASA PCoE Li-ion Battery Data Set, following the step-by-step process discussed earlier: load raw cycling data, extract discharge-curve features, label SOH and RUL as classes, train/test split by battery (not by cycle), train several classifiers, and evaluate.

See `outputs/results_summary.md` for the actual results and what they mean for your project report.

## Structure

```
smartcell_soh_rul/
├── data/nasa_mat/          B0005.mat, B0006.mat, B0007.mat, B0018.mat (raw NASA data)
├── scripts/
│   ├── soh_rul_pipeline.py     Entry point - run this
│   └── smartcell_ml/
│       ├── config.py           All settings: paths, thresholds, feature list, model hyperparams
│       ├── data.py             Load .mat files, extract features, build SOH/RUL labels
│       ├── evaluate.py         Leave-one-battery-out training + metrics
│       ├── plots.py            Confusion matrices, capacity-fade curve, feature importance
│       └── pipeline.py         Wires the above together, writes outputs/
├── outputs/
│   ├── engineered_features.csv   Full feature table (636 discharge cycles × 14 features + labels)
│   ├── results.json               All metrics, confusion matrices, classification reports, feature importances
│   ├── results_summary.md         Written summary of methodology and results (report-ready)
│   └── plots/                     capacity_fade_soh.png, confusion_matrices_{SOH,RUL}.png, feature_importance_{SOH,RUL}.png
├── README.md
└── requirements.txt
```

## Running it yourself

```
pip install -r requirements.txt
python3 scripts/soh_rul_pipeline.py
```

Takes under a minute on a laptop. It regenerates everything in `outputs/`.

## Data source

The `.mat` files are the original NASA Prognostics Center of Excellence Battery Data Set (B. Saha and K. Goebel, 2007), room-temperature aging experiments on four Li-ion cells cycled to failure. Official repository: NASA's PCoE Data Set Repository (mirrored by the PHM Society at data.phmsociety.org/nasa). If you need to re-download the raw files yourself and the NASA/PHM mirrors are unreachable, they're also available (identical files, checked against the original) via the `Maruthi-prasanth/Battery-Health-Prophecy` GitHub repository.

## Why this is a proof-of-concept, not the deployed model

Per the Revised Directions document responding to your Review 0 panel feedback, the ESP32 pipeline should use a classical, explainable SOH/RUL estimator (internal resistance trend, Coulomb counting, OCV lookup) rather than an ML/LSTM layer. This pipeline exists to validate that decision with data: it shows (a) that a classification approach on discharge-curve features works well even generalizing to unseen batteries, and (b) that the features actually driving the classification — time-to-cutoff-voltage and discharge energy — are exactly the kind of signal your classical estimator is already built on. It uses a public Li-ion dataset because no equivalent public dataset exists for 12V lead-acid batteries in two-wheeler use; treat it as validating the *method*, and plan to re-run the same feature-extraction-and-classification pipeline once you have real lead-acid cycling data from your own hardware.
