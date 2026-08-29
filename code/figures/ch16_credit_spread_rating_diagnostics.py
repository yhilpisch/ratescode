"""
Python & AI for Rates, Bonds, and Credit
Chapter 16 · Credit Spreads, Ratings, and Default Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit spread and rating diagnostics figure for Chapter 16.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create the credit spread and rating diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(DATA / "ch16_synthetic_issuer_panel.csv")
    spreads = pd.read_csv(DATA / "ch16_credit_spread_panel.csv")
    panel["expected_loss_bp"] = panel["pd"] * (1 - panel["recovery"]) * 10000

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    rating_avg = panel.groupby("rating")["spread_bp"].mean()
    axes[0].bar(rating_avg.index, rating_avg.values, color=BLUE)
    axes[0].set_title("Average spread by rating", color=NAVY)
    axes[0].set_ylabel("Spread bp")

    x = range(len(panel))
    axes[1].bar(x, panel["spread_bp"], color=BLUE, label="spread")
    axes[1].bar(x, panel["expected_loss_bp"], color=ORANGE,
                label="expected loss")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(panel["rating"])
    axes[1].set_title("Spread versus expected loss", color=NAVY)
    axes[1].set_ylabel("Basis points")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch16_credit_spread_rating_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
