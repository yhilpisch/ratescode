"""
Python & AI for Rates, Bonds, and Credit
Chapter 5 · Money Markets, Short Rates, and Overnight Indexing

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Overnight compounding path figure for Chapter 5.
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
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create a figure for compounded overnight index growth."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    rates = pd.read_csv(DATA / "ch05_overnight_rates.csv")
    days = rates["day"].to_numpy(int)
    overnight = rates["overnight_rate"].to_numpy(float)
    tau = np.full_like(overnight, 1.0 / 360.0)  # ACT/360 daily fractions
    compounded = np.cumprod(1.0 + overnight * tau) - 1.0
    simple = np.cumsum(overnight * tau)

    excess_bp = (compounded - simple) * 10000

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 3.6), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    axes[0].plot(
        days, compounded * 10000, color=NAVY, marker="o", lw=2.2,
        label="compounded",
    )
    axes[0].plot(
        days, simple * 10000, "--", color=ORANGE, marker="s", lw=1.8,
        label="simple accumulation",
    )
    axes[0].set_title("Daily overnight accrual", color=NAVY)
    axes[0].set_ylabel("Cumulative return (bp)")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(days, excess_bp, color=BLUE, marker="o", lw=2.0)
    axes[1].axhline(0.0, color=NAVY, lw=0.8)
    axes[1].set_title("Compounding excess", color=NAVY)
    axes[1].set_xlabel("Accrual day")
    axes[1].set_ylabel("Excess (bp)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch05_overnight_compounding_path.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
