"""Leave-one-battery-out training and evaluation for a given target label column."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from . import config


def evaluate_target(df: pd.DataFrame, target_col: str, label_order: list) -> dict:
    """Train/test every model in config.MODELS with each battery held out in turn.

    Returns {model_name: {accuracy, f1_weighted, precision_weighted, recall_weighted,
    confusion_matrix, labels, classification_report}}.
    """
    X = df[config.FEATURE_COLS].values
    y = df[target_col].astype(str).values
    groups = df["battery_id"].values

    logo = LeaveOneGroupOut()
    predictions = {name: {"y_true": [], "y_pred": []} for name in config.MODELS}

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        for name, make_model in config.MODELS.items():
            model = make_model()
            # Logistic Regression is distance-based, so it needs scaled features;
            # the tree-based models are scale-invariant and use the raw values.
            if name == "LogisticRegression":
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            predictions[name]["y_true"].extend(y_test.tolist())
            predictions[name]["y_pred"].extend(pred.tolist())

    summary = {}
    for name, p in predictions.items():
        y_true, y_pred = p["y_true"], p["y_pred"]
        summary[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_order).tolist(),
            "labels": label_order,
            "classification_report": classification_report(
                y_true, y_pred, labels=label_order, zero_division=0
            ),
        }
    return summary
