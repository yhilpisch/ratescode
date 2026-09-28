"""
Python & AI for Rates, Bonds, and Credit
Chapter 26 · ML for Credit Spreads and Default Risk

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
FEATURES = ["spread_bp", "spread_mom", "leverage", "coverage", "macro"]


def load_panel() -> pd.DataFrame:
    """Load the frozen synthetic credit ML panel."""
    return pd.read_csv(DATA / "ch26_credit_ml_panel.csv")


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return logistic probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def chronological_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier issuer dates and hold out the final observed date."""
    panel = load_panel()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel = panel.sort_values(["date", "issuer"]).reset_index(drop=True)
    dates = panel["date"].drop_duplicates().sort_values()
    if len(dates) < 2:
        raise ValueError("at least two observation dates are required")
    cutoff = dates.iloc[-1]
    train = panel.loc[panel["date"] < cutoff].copy()
    test = panel.loc[panel["date"] == cutoff].copy()
    return train, test


def fit_logistic(
    train: pd.DataFrame,
    steps: int=100,
    lr: float=0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit logistic regression and scaling parameters on training rows only."""
    if steps < 1 or lr <= 0.0:
        raise ValueError("steps and learning rate must be positive")
    x = train[FEATURES].to_numpy(float)
    y = train["label"].to_numpy(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-12
    x_scaled = np.c_[np.ones(len(x)), (x - mean) / scale]
    beta = np.zeros(x_scaled.shape[1])
    for _ in range(steps):
        prediction = sigmoid(x_scaled @ beta)
        gradient = x_scaled.T @ (prediction - y) / len(y)
        beta -= lr * gradient
    return beta, mean, scale


def classification_table() -> pd.DataFrame:
    """Return final-date holdout predictions for the synthetic credit panel."""
    train, test = chronological_split()
    beta, mean, scale = fit_logistic(train)
    x = test[FEATURES].to_numpy(float)
    x_scaled = np.c_[np.ones(len(x)), (x - mean) / scale]
    test["probability"] = sigmoid(x_scaled @ beta)
    test["prediction"] = (test["probability"] >= 0.5).astype(int)
    return test[["date", "issuer", "rating", "label", "probability",
                 "prediction"]]


def metrics() -> pd.DataFrame:
    """Compute descriptive metrics on the final-date synthetic holdout."""
    table = classification_table()
    tp = int(((table["prediction"] == 1) & (table["label"] == 1)).sum())
    fp = int(((table["prediction"] == 1) & (table["label"] == 0)).sum())
    fn = int(((table["prediction"] == 0) & (table["label"] == 1)).sum())
    accuracy = float((table["prediction"] == table["label"]).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return pd.DataFrame([{
        "test_observations": len(table),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }])


def main() -> None:
    """Print a compact chapter result summary."""
    print(classification_table().round({"probability": 3}))
    print(metrics().round(3))


if __name__ == "__main__":
    main()
