"""
Python & AI for Rates, Bonds, and Credit
Chapter 25 · ML for Yield Curves and Rates

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
YIELD_COLS = ["y1", "y2", "y5", "y10"]


def curve_features() -> pd.DataFrame:
    """Create curve-factor features from frozen yield nodes."""
    panel = pd.read_csv(DATA / "ch25_curve_ml_panel.csv")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel = panel.sort_values("date").reset_index(drop=True)
    if panel["date"].duplicated().any():
        raise ValueError("the teaching panel requires unique observation dates")
    panel["level"] = panel[YIELD_COLS].mean(axis=1)
    panel["slope"] = panel["y10"] - panel["y1"]
    panel["curvature"] = 2 * panel["y5"] - panel["y2"] - panel["y10"]
    panel["target"] = panel["y10"].diff().shift(-1)  # next 10Y change
    panel["target_end_date"] = panel["date"].shift(-1)
    return panel.dropna().reset_index(drop=True)


def chronological_split(
    first_test_row: int=7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date and purge training labels reaching the test cutoff."""
    frame = curve_features()
    if not 1 <= first_test_row < len(frame):
        raise ValueError("first_test_row must leave training and test rows")
    test = frame.iloc[first_test_row:].copy()
    test_start = test["date"].iloc[0]
    train = frame.iloc[:first_test_row].copy()
    train = train.loc[train["target_end_date"] < test_start].copy()
    if train.empty or test.empty:
        raise ValueError("the split must leave non-empty train/test samples")
    return train, test


def ridge_forecast(alpha: float=1.0) -> pd.DataFrame:
    """Fit on earlier dates and forecast the held-out synthetic observations."""
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    train, test = chronological_split()
    features = ["level", "slope", "curvature", "macro", "volatility"]
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train["target"].to_numpy(float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) + 1e-12
    x_train = np.c_[np.ones(len(x_train)), (x_train - mean) / scale]
    x_test = np.c_[np.ones(len(x_test)), (x_test - mean) / scale]
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    test["ridge"] = x_test @ beta
    test["random_walk"] = 0.0  # no-change baseline
    return test[["date", "target", "random_walk", "ridge"]]


def regime_summary() -> pd.DataFrame:
    """Summarise frozen curve regimes."""
    regimes = pd.read_csv(DATA / "ch25_curve_regimes.csv")
    return regimes.groupby("regime")[["level", "slope"]].mean().reset_index()


def main() -> None:
    """Print a compact chapter result summary."""
    print(ridge_forecast().round(4))
    print(regime_summary().round(4))


if __name__ == "__main__":
    main()
