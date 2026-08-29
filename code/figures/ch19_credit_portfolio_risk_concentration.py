"""
Python & AI for Rates, Bonds, and Credit
Chapter 19 · Credit Portfolios and Spread-Risk Management

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit portfolio risk concentration figure for Chapter 19.
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
    """Create the credit portfolio concentration figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    portfolio = pd.read_csv(DATA / "ch19_credit_portfolio.csv")
    portfolio["cs01"] = portfolio["market_value"]
    portfolio["cs01"] *= portfolio["spread_duration"] / 10000.0
    sector = portfolio.groupby("sector")["market_value"].sum()
    rating = portfolio.groupby("rating")["cs01"].sum()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].barh(sector.index, sector.values / 1_000_000, color=BLUE)
    axes[0].set_title("Sector exposure", color=NAVY)
    axes[0].set_xlabel("Market value million")

    axes[1].bar(rating.index, rating.values, color=ORANGE)
    axes[1].set_title("CS01 by rating", color=NAVY)
    axes[1].set_ylabel("CS01")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch19_credit_portfolio_risk_concentration.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
