"""
Python & AI for Rates, Bonds, and Credit
Chapter 25 · ML for Yield Curves and Rates

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Curve ML regime diagnostics figure for Chapter 25.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
YIELD_COLS = ["y1", "y2", "y5", "y10"]


def ridge_forecast(alpha: float=1.0) -> pd.DataFrame:
    """Fit a ridge model to next-period 10Y yield changes."""
    panel = pd.read_csv(DATA / "ch25_curve_ml_panel.csv")
    panel["level"] = panel[YIELD_COLS].mean(axis=1)
    panel["slope"] = panel["y10"] - panel["y1"]
    panel["curvature"] = 2 * panel["y5"] - panel["y2"] - panel["y10"]
    panel["target"] = panel["y10"].diff().shift(-1)
    frame = panel.dropna().reset_index(drop=True)
    features = ["level", "slope", "curvature", "macro", "volatility"]
    train = frame.iloc[:7]
    test = frame.iloc[7:].copy()
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
    test["random_walk"] = 0.0
    return test[["date", "target", "random_walk", "ridge"]]


def main() -> None:
    """Create the curve ML regime figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    regimes = pd.read_csv(DATA / "ch25_curve_regimes.csv")
    colors = {"calm": BLUE, "transition": LIGHT_BLUE, "stress": ORANGE}
    color_values = [colors[item] for item in regimes["regime"]]
    forecast = ridge_forecast()
    error = pd.DataFrame({
        "model": ["random walk", "ridge"],
        "mae": [
            (forecast["target"] - forecast["random_walk"]).abs().mean(),
            (forecast["target"] - forecast["ridge"]).abs().mean(),
        ],
    })

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].scatter(regimes["slope"], regimes["level"], c=color_values, s=70)
    axes[0].set_title("Curve regimes", color=NAVY)
    axes[0].set_xlabel("Slope")
    axes[0].set_ylabel("Level")

    axes[1].bar(error["model"], error["mae"], color=[BLUE, ORANGE])
    axes[1].set_title("Forecast error", color=NAVY)
    axes[1].set_ylabel("MAE")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch25_curve_ml_regimes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
