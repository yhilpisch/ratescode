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

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(days, compounded * 10000, color=NAVY, lw=2.2, label="compounded")
    ax.plot(days, simple * 10000, "--", color=ORANGE, lw=1.8, label="simple")
    daily_accrual = overnight * tau * 10000  # daily accrual in basis points
    ax.bar(days, daily_accrual, color=BLUE, alpha=0.18, label="daily accrual")

    ax.set_title("Overnight compounding accumulates day by day", color=NAVY)
    ax.set_xlabel("Accrual day")
    ax.set_ylabel("Cumulative return in basis points")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch05_overnight_compounding_path.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
