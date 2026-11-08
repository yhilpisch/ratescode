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
    fig, (ax, ax_error) = plt.subplots(
        2, 1, figsize=(7.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    x = shifts * 10000
    ax.plot(x, exact, color=NAVY, lw=2.6, label="exact price")
    ax.plot(
        x,
        duration_only,
        "--",
        color=ORANGE,
        lw=2.0,
        label="duration only",
    )
    ax.plot(
        x,
        duration_convexity,
        ":",
        color=GREEN,
        lw=2.6,
        label="duration plus convexity",
    )

    ax.set_title("Convexity materially reduces large-shock pricing error",
                 color=NAVY)
    ax.set_ylabel("Bond price")
    duration_error = np.abs(duration_only - exact) * 100
    convexity_error = np.abs(duration_convexity - exact) * 100
    duration_error = np.where(duration_error > 1e-6, duration_error, np.nan)
    convexity_error = np.where(
        convexity_error > 1e-6, convexity_error, np.nan
    )
    ax_error.plot(
        x, duration_error,
        "--", color=ORANGE, lw=2.0, label="duration error",
    )
    ax_error.plot(
        x, convexity_error,
        ":", color=GREEN, lw=2.6, label="with convexity",
    )
    ax_error.set_xlabel("Yield shift (basis points)")
    ax_error.set_ylabel("Absolute error (cents, log scale)")
    ax_error.set_yscale("log")
    ax_error.grid(True, which="both", axis="y")
    fig.legend(
        handles=[ax.lines[0], ax.lines[1], ax.lines[2]],
        labels=["exact price", "duration only", "duration + convexity"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax_error.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax_error.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax_error.xaxis.label.set_color(NAVY)
    ax_error.yaxis.label.set_color(NAVY)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "ch03_duration_convexity_approximation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
