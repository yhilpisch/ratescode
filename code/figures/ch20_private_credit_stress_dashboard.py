"""
Python & AI for Rates, Bonds, and Credit
Chapter 20 · Private Credit Analytics

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Private credit stress dashboard figure for Chapter 20.
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
    """Create the private credit stress dashboard figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    loans = pd.read_csv(DATA / "ch20_private_credit_loans.csv")
    covenants = pd.read_csv(DATA / "ch20_borrower_metrics.csv")
    book = loans.merge(covenants, on="borrower")
    book["leverage"] = book["debt"] / book["ebitda"]
    book["coverage"] = book["ebitda"] / book["interest"]
    stress = pd.read_csv(DATA / "ch20_private_credit_stress_scenarios.csv")
    values = []
    for scenario in stress.itertuples():
        rate = book["base_rate"] + scenario.base_rate_shock + book["spread"]
        ebitda = book["ebitda"] * (1.0 + scenario.ebitda_shock)
        coverage = ebitda / (book["principal"] * rate)
        values.append(float(coverage.min()))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(book["borrower"], book["leverage"], color=BLUE)
    axes[0].set_title("Borrower leverage", color=NAVY)
    axes[0].set_ylabel("Debt/EBITDA")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(stress["scenario"], values, color=ORANGE)
    axes[1].set_title("Minimum stress coverage", color=NAVY)
    axes[1].set_ylabel("Coverage")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch20_private_credit_stress_dashboard.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
