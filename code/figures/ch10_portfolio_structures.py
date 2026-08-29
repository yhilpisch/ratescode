"""
Python & AI for Rates, Bonds, and Credit
Chapter 10 · Bond Portfolio Construction and Risk Budgets

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Portfolio structure and DV01 figure for Chapter 10.
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
LIGHT_BLUE = "#8CB6E8"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def profile_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return portfolio names, allocation weights, and DV01 values."""
    profiles = pd.read_csv(DATA / "ch10_portfolio_profiles.csv")
    bucket = pd.read_csv(DATA / "ch10_bond_portfolio.csv")
    names = profiles["portfolio"].tolist()
    weights = profiles[["short", "belly", "long"]].to_numpy(float)
    bucket_dv01 = bucket["dv01"].to_numpy(float)
    dv01 = weights @ bucket_dv01
    return names, weights, dv01


def main() -> None:
    """Create the portfolio structure figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    names, weights, dv01 = profile_data()
    x = np.arange(len(names))  # portfolio locations
    colors = [LIGHT_BLUE, BLUE, ORANGE]  # bucket colors
    labels = ["short", "belly", "long"]  # maturity buckets

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))

    bottom = np.zeros(len(names))  # stacked-bar baseline
    for idx, label in enumerate(labels):
        axes[0].bar(
            x,
            weights[:, idx],
            bottom=bottom,
            color=colors[idx],
            label=label,
        )
        bottom += weights[:, idx]

    axes[0].set_title("Maturity allocation", color=NAVY)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Portfolio weight")
    axes[0].legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    axes[1].bar(names, dv01, color=BLUE)
    axes[1].set_title("Resulting DV01", color=NAVY)
    axes[1].set_ylabel("DV01 per 10 million")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch10_portfolio_structures.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
