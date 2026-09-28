"""
Python & AI for Rates, Bonds, and Credit
Chapter 27 · ML for Portfolio and Risk Signals

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Portfolio risk signal figure for Chapter 27.
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
    """Create the portfolio risk signal figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(DATA / "ch27_portfolio_signal_panel.csv")
    alerts = pd.read_csv(DATA / "ch27_stress_alerts.csv")
    panel["date"] = pd.to_datetime(panel["date"])
    alerts["date"] = pd.to_datetime(alerts["date"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(panel["date"], panel["loss"], color=BLUE, label="loss")
    axes[0].scatter(alerts["date"], [panel["loss"].max()] * len(alerts),
                    color=ORANGE, marker="v", label="alerts")
    axes[0].set_title("Loss path and alerts", color=NAVY)
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(alerts["date"], alerts["score"], color=ORANGE,
                 marker="o")
    axes[1].set_title("Fixed logistic-style score", color=NAVY)
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)

    for ax in axes:
        ax.tick_params(colors=NAVY, axis="x", rotation=25)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch27_portfolio_risk_signals.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
