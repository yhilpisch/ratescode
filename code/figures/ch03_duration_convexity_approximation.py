"""
Python & AI for Rates, Bonds, and Credit
Chapter 3 · Bond Mathematics and Market Conventions

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Duration and convexity approximation figure for Chapter 3.
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
GREEN = "#2E8B57"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def bond_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return frozen coupon times and cash flows."""
    table = pd.read_csv(DATA / "ch03_bond_cashflows.csv")
    return (
        table["time"].to_numpy(float),
        table["cashflow"].to_numpy(float),
    )


def bond_price(yield_pa: float) -> float:
    """Price a stylised semi-annual coupon bond from a flat yield."""
    times, cashflows = bond_inputs()
    discounts = (1.0 + yield_pa / 2.0) ** (-2.0 * times)
    return float(np.sum(cashflows * discounts))


def risk_measures(yield_pa: float) -> tuple[float, float, float]:
    """Return price, modified duration, and convexity."""
    times, cashflows = bond_inputs()
    periods = np.arange(1, len(times) + 1)
    base = 1.0 + yield_pa / 2.0  # periodic discount base
    pv = cashflows * base ** (-periods)  # present-value contributions
    price = float(np.sum(pv))
    macaulay = float(np.dot(times, pv) / price)
    modified = macaulay / base
    convexity = float(np.sum(pv * periods * (periods + 1)))
    convexity /= price * 2.0**2 * base**2
    return price, modified, convexity


def main() -> None:
    """Create the duration-convexity approximation figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    base_yield = 0.04
    base_price, duration, convexity = risk_measures(base_yield)
    shifts = np.linspace(-0.02, 0.02, 81)  # yield shifts in decimal form
    yields = base_yield + shifts

    exact = np.array([bond_price(y) for y in yields])
    duration_only = base_price * (1.0 - duration * shifts)
    duration_convexity = base_price * (
        1.0 - duration * shifts + 0.5 * convexity * shifts**2
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(shifts * 10000, exact, color=NAVY, lw=2.2, label="exact price")
    ax.plot(
        shifts * 10000,
        duration_only,
        "--",
        color=ORANGE,
        lw=1.8,
        label="duration only",
    )
    ax.plot(
        shifts * 10000,
        duration_convexity,
        ":",
        color=GREEN,
        lw=2.2,
        label="duration plus convexity",
    )

    ax.set_title("Duration and convexity as local approximations", color=NAVY)
    ax.set_xlabel("Yield shift in basis points")
    ax.set_ylabel("Bond price")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch03_duration_convexity_approximation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
