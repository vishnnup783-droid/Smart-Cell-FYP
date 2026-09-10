"""Orchestrates the full run: load data -> evaluate SOH -> evaluate RUL -> save everything."""

import json
import warnings

from . import config
from .data import build_dataset
from .evaluate import evaluate_target
from .plots import plot_capacity_fade, plot_confusion_matrices, plot_feature_importance

warnings.filterwarnings("ignore")


def _add_target_report(report: dict, prefix: str, summary: dict, importance) -> None:
    """Folds one target's (SOH or RUL) results into the shared report dict."""
    report[f"{prefix}_results"] = {
        name: {k: v for k, v in res.items() if k != "classification_report"}
        for name, res in summary.items()
    }
    report[f"{prefix}_classification_reports"] = {
        name: res["classification_report"] for name, res in summary.items()
    }
    report[f"{prefix}_feature_importance_rf"] = importance.to_dict()


def _run_target(df, target_col: str, label_order: list, target_name: str, report: dict):
    print(f"\nEvaluating {target_name} classifiers (leave-one-battery-out) ...")
    summary = evaluate_target(df, target_col, label_order)
    plot_confusion_matrices(summary, target_name)
    importance = plot_feature_importance(df, target_col, target_name)
    _add_target_report(report, target_name.lower(), summary, importance)

    print(f"\n===== {target_name} classification (leave-one-battery-out) =====")
    for name, res in summary.items():
        print(f"{name:20s} acc={res['accuracy']:.3f}  f1_weighted={res['f1_weighted']:.3f}")
    print(f"\nTop {target_name} features (Random Forest):")
    print(importance.head(5))
    return summary, importance


def main():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and parsing NASA .mat files ...")
    df = build_dataset()
    print(f"Built dataset: {len(df)} discharge cycles across {df['battery_id'].nunique()} batteries")
    print(df.groupby("battery_id").size())

    df.to_csv(config.OUT_DIR / "engineered_features.csv", index=False)
    print(f"Saved engineered feature table -> {config.OUT_DIR / 'engineered_features.csv'}")

    plot_capacity_fade(df)

    report = {
        "rated_capacity_ah": config.RATED_CAPACITY_AH,
        "eol_fraction": config.EOL_FRACTION,
        "soh_bins_pct": config.SOH_BINS,
        "soh_labels": config.SOH_LABELS,
        "rul_bin_edges_cycles": df.attrs["rul_bin_edges"],
        "rul_labels": config.RUL_LABELS,
        "n_discharge_cycles": len(df),
        "class_balance": {
            "soh_class": df["soh_class"].value_counts().to_dict(),
            "rul_class": df["rul_class"].value_counts().to_dict(),
        },
    }

    _run_target(df, "soh_class", config.SOH_LABELS, "SOH", report)
    _run_target(df, "rul_class", config.RUL_LABELS, "RUL", report)

    with open(config.OUT_DIR / "results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full metrics -> {config.OUT_DIR / 'results.json'}")

    return df, report


if __name__ == "__main__":
    main()
