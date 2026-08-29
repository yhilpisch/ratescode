"""
Python & AI for Rates, Bonds, and Credit
Chapter 23 · Model Risk, Calibration Risk, and Verification

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Calibration and verification diagnostics figure for Chapter 23.
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


def residual_data() -> pd.DataFrame:
    """Return residuals for two curve calibration choices."""
    quotes = pd.read_csv(DATA / "ch23_curve_quotes.csv")
    quotes["linear"] = (quotes["model_linear"] - quotes["market_rate"])
    quotes["smooth"] = (quotes["model_smooth"] - quotes["market_rate"])
    quotes["linear_bp"] = quotes["linear"] * 10000.0
    quotes["smooth_bp"] = quotes["smooth"] * 10000.0
    return quotes


def dv01_diagnostic() -> tuple[float, float]:
    """Return finite-difference and approximation DV01 values."""
    flows = pd.read_csv(DATA / "ch23_bond_roundtrip.csv")
    times = flows["time"].to_numpy(float)
    cashflows = flows["cashflow"].to_numpy(float)
    ytm = 0.045  # base yield
    discount = (1.0 + ytm / 2) ** (-2 * times)
    price = float(np.dot(cashflows, discount))
    bumped = (1.0 + (ytm + 0.0001) / 2) ** (-2 * times)
    bumped_price = float(np.dot(cashflows, bumped))
    fd_dv01 = price - bumped_price
    weights = cashflows * discount / price  # value weights
    duration = float(np.dot(times, weights)) / (1.0 + ytm / 2)
    approx = price * duration / 10000.0
    return fd_dv01, approx


def main() -> None:
    """Create the calibration and verification diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    residuals = residual_data()
    fd_dv01, approx_dv01 = dv01_diagnostic()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    axes[0].plot(
        residuals["maturity"],
        residuals["linear_bp"],
        marker="o",
        color=BLUE,
        label="linear",
    )
    axes[0].plot(
        residuals["maturity"],
        residuals["smooth_bp"],
        marker="s",
        color=ORANGE,
        label="smooth",
    )
    axes[0].axhline(0, color=NAVY, lw=1)
    axes[0].set_title("Calibration residuals", color=NAVY)
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Residual bp")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(["finite diff", "duration"], [fd_dv01, approx_dv01],
                color=[BLUE, ORANGE])
    axes[1].set_title("DV01 verification", color=NAVY)
    axes[1].set_ylabel("DV01")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch23_calibration_verification_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
