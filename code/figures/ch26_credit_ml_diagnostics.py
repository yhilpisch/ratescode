"""
Python & AI for Rates, Bonds, and Credit
Chapter 26 · ML for Credit Spreads and Default Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit ML diagnostics figure for Chapter 26.
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
    """Create the credit ML diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(DATA / "ch26_credit_classification_metrics.csv")
    panel = pd.read_csv(DATA / "ch26_credit_ml_panel.csv")
    importance = panel[["spread_mom", "leverage", "coverage", "macro"]]
    importance = importance.corrwith(panel["label"]).abs().sort_values()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(metrics["model"], metrics["recall"], color=BLUE)
    axes[0].set_title("Stress recall", color=NAVY)
    axes[0].set_ylabel("Recall")

    axes[1].barh(importance.index, importance.values, color=ORANGE)
    axes[1].set_title("Feature association", color=NAVY)
    axes[1].set_xlabel("Absolute correlation")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch26_credit_ml_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
