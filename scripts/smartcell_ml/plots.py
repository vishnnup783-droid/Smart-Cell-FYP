"""All matplotlib output: confusion matrices, capacity-fade trajectory, feature importance."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def plot_confusion_matrices(summary: dict, target_name: str):
    fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4))
    for ax, (name, res) in zip(axes, summary.items()):
        cm = np.array(res["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nacc={res['accuracy']:.2f}  f1={res['f1_weighted']:.2f}")
        ax.set_xticks(range(len(res["labels"])))
        ax.set_yticks(range(len(res["labels"])))
        ax.set_xticklabels(res["labels"], rotation=45, ha="right")
        ax.set_yticklabels(res["labels"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, cm[r, c], ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.suptitle(f"Leave-one-battery-out confusion matrices - {target_name}")
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / f"confusion_matrices_{target_name}.png", dpi=150)
    plt.close(fig)


def plot_capacity_fade(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in config.BATTERIES:
        sub = df[df["battery_id"] == b].sort_values("discharge_cycle")
        ax.plot(sub["discharge_cycle"], sub["soh_pct"], label=b)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% (Healthy/Degrading)")
    ax.axhline(65, color="gray", linestyle=":", linewidth=1, label="65% (Degrading/Critical)")
    ax.axhline(70, color="red", linestyle="--", linewidth=1, label="70% EOL threshold")
    ax.set_xlabel("Discharge cycle number")
    ax.set_ylabel("SOH (%)")
    ax.set_title("Capacity fade / SOH trajectory - NASA batteries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "capacity_fade_soh.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, target_col: str, target_name: str) -> pd.Series:
    """Fits a fresh Random Forest on the full dataset (not CV) just to rank features."""
    X = df[config.FEATURE_COLS].values
    y = df[target_col].astype(str).values
    model = config.MODELS["RandomForest"]()
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=config.FEATURE_COLS).sort_values()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    importances.plot.barh(ax=ax, color="#3b6fa0")
    ax.set_title(f"Random Forest feature importance - {target_name} (full-data fit)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / f"feature_importance_{target_name}.png", dpi=150)
    plt.close(fig)
    return importances.sort_values(ascending=False)
