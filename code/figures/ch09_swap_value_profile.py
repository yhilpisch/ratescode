"""
Python & AI for Rates, Bonds, and Credit
Chapter 9 · Interest-Rate Swaps and Swap Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Swap value profile figure for Chapter 9.
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


def swap_inputs() -> tuple[float, float, float]:
    """Return annuity, par rate, and notional for a toy swap."""
    table = pd.read_csv(DATA / "ch09_swap_inputs.csv")
    discounts = table["discount"].to_numpy(float)
    forwards = table["forward"].to_numpy(float)
    alpha = table["alpha"].to_numpy(float)
    annuity = float(np.sum(alpha * discounts))  # fixed-leg annuity
    floating_pv = float(np.sum(discounts * alpha * forwards))
    par_rate = floating_pv / annuity  # par fixed rate
    notional = 50_000_000.0  # swap notional
    return annuity, par_rate, notional


def main() -> None:
    """Create the swap value profile figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    annuity, par_rate, notional = swap_inputs()
    fixed_rates = np.linspace(par_rate - 0.006, par_rate + 0.006, 81)
    values = notional * annuity * (fixed_rates - par_rate)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(fixed_rates * 100, values / 1_000_000, color=BLUE, lw=2.2)
    ax.axhline(0.0, color=NAVY, lw=1.0)
    ax.axvline(par_rate * 100, color=ORANGE, lw=1.8, ls="--")
    ax.text(
        par_rate * 100 + 0.03,
        0.35,
        "par rate",
        color=ORANGE,
        fontsize=9,
    )

    ax.set_title("Receiver swap value increases with fixed rate", color=NAVY)
    ax.set_xlabel("Contract fixed rate in percent")
    ax.set_ylabel("Receiver value in millions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch09_swap_value_profile.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
