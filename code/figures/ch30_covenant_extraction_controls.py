"""
Python & AI for Rates, Bonds, and Credit
Chapter 30 · Credit Documents, Covenants, and Research Notes

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Covenant extraction controls figure for Chapter 30.
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
    """Create the covenant extraction controls figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(DATA / "ch30_covenant_extractions.csv")
    risk = pd.read_csv(DATA / "ch30_credit_risk_summary.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(cov["issuer"], cov["extracted_value"], color=BLUE)
    axes[0].scatter(cov["issuer"], cov["threshold"], color=ORANGE,
                    zorder=3, label="threshold")
    axes[0].set_title("Extracted covenant values", color=NAVY)
    axes[0].set_ylabel("Ratio (x)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(risk["issuer"], risk["headroom"], color=ORANGE)
    axes[1].set_title("Covenant headroom", color=NAVY)
    axes[1].set_ylabel("Headroom (x)")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch30_covenant_extraction_controls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
