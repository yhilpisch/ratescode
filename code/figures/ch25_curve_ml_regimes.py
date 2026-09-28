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
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "code"))
from ch25_ml_yield_curves_rates import ridge_forecast


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
    axes[1].set_title("Small-sample error illustration", color=NAVY)
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
