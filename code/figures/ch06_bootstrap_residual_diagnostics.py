"""
Python & AI for Rates, Bonds, and Credit
Chapter 6 · Curve Bootstrapping and Multi-Curve Discounting

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Bootstrap and residual diagnostics figure for Chapter 6.
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


def bootstrap_discounts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap annual discount factors from a toy swap curve."""
    table = pd.read_csv(DATA / "ch06_swap_rates.csv")
    maturities = table["maturity"].to_numpy(float)
    swap_rates = table["swap_rate"].to_numpy(float)
    discounts: list[float] = []

    for n, rate in enumerate(swap_rates, start=1):
        known = sum(discounts)  # annual fixed coupons already discounted
        next_discount = (1.0 - rate * known) / (1.0 + rate)
        discounts.append(float(next_discount))

    return maturities, swap_rates, np.array(discounts)


def par_swap_rate(discounts: np.ndarray, n_periods: int) -> float:
    """Compute a par swap rate from annual discount factors."""
    annuity = float(np.sum(discounts[:n_periods]))
    return (1.0 - float(discounts[n_periods - 1])) / annuity


def main() -> None:
    """Create the bootstrap diagnostic figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    maturities, quotes, discounts = bootstrap_discounts()
    fitted = np.array(
        [par_swap_rate(discounts, n) for n in range(1, len(discounts) + 1)]
    )
    residuals_bp = (fitted - quotes) * 10000.0  # residuals in basis points
    residuals_bp[np.abs(residuals_bp) < 1e-10] = 0.0  # remove round-off noise

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    axes[0].plot(maturities, discounts, marker="o", color=NAVY, lw=2.0)
    axes[0].set_title("Bootstrapped discount factors", color=NAVY)
    axes[0].set_xlabel("Maturity in years")
    axes[0].set_ylabel("Discount factor")

    axes[1].bar(maturities, residuals_bp, color=ORANGE, width=0.35)
    axes[1].axhline(0.0, color=NAVY, lw=1.0)
    axes[1].set_title("Quote repricing residuals", color=NAVY)
    axes[1].set_xlabel("Maturity in years")
    axes[1].set_ylabel("Residual in basis points")
    axes[1].set_ylim(-0.02, 0.02)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch06_bootstrap_residual_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
