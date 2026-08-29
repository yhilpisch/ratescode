"""
Python & AI for Rates, Bonds, and Credit
Chapter 24 · Machine Learning Workflow for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ["level", "slope", "curvature", "spread", "volatility", "macro"]


def load_panel() -> pd.DataFrame:
    """Load the frozen ML feature panel."""
    return pd.read_csv(DATA / "ch24_ml_feature_panel.csv")


def time_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a simple chronological train/test split."""
    panel = load_panel()
    split = int(len(panel) * 0.70)  # chronological split point
    return panel.iloc[:split].copy(), panel.iloc[split:].copy()


def ridge_predict(alpha: float=1.0) -> pd.DataFrame:
    """Fit a small ridge regression and return test predictions."""
    train, test = time_split()
    x_train = train[FEATURES].to_numpy(float)
    y_train = train["target"].to_numpy(float)
    x_test = test[FEATURES].to_numpy(float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) + 1e-12  # stable scaling
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0  # do not penalise intercept
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    test["prediction"] = x_test @ beta
    return test[["date", "target", "prediction"]]


def model_comparison() -> pd.DataFrame:
    """Compare baseline and ridge forecasts on the test sample."""
    train, test = time_split()
    predictions = ridge_predict()
    baseline = float(train["target"].mean())  # no-skill forecast
    rows = [
        {
            "model": "baseline",
            "mae": float((test["target"] - baseline).abs().mean()),
            "hit_rate": float((np.sign(test["target"]) == np.sign(baseline))
                              .mean()),
        },
        {
            "model": "ridge",
            "mae": float((predictions["target"]
                          - predictions["prediction"]).abs().mean()),
            "hit_rate": float((np.sign(predictions["target"])
                               == np.sign(predictions["prediction"])).mean()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(ridge_predict().round(4))
    print(model_comparison().round(4))


if __name__ == "__main__":
    main()
