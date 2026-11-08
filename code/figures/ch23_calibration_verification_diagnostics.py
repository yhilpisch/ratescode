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
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from credit_figure_layout import (
    NAVY, BLUE, ORANGE, export_panels, legend_above,
)

from ch23_model_risk_calibration_risk_verification import (
    calibration_residuals, finite_difference_dv01, price_from_yield,
    yield_from_price,
)


def draw_residuals(ax) -> None:
    """Plot supplied candidate residuals without implying a fitted solver."""
    table = calibration_residuals()
    for column, color, name in [
        ("linear_residual_bp", BLUE, "Linear"),
        ("smooth_residual_bp", ORANGE, "Smooth"),
    ]:
        rms = np.sqrt(np.mean(table[column] ** 2))
        ax.plot(table.maturity, table[column], marker="o", color=color,
                lw=2, label=f"{name}: RMS {rms:.2f} bp")
    ax.axhline(0, color=NAVY, lw=1)
    ax.set(xlabel="Quote maturity (years)", ylabel="Model - target (bp)")
    legend_above(ax)


def bump_comparison() -> pd.DataFrame:
    """Compare derivative approximations against the analytical local DV01."""
    y, scale = 0.045, 1e-4
    base = price_from_yield(y)
    reference = finite_difference_dv01()["approx_dv01"]
    rows = []
    for bp in [0.1, 0.25, 0.5, 1, 2, 5, 10]:
        h = bp * scale
        plus = price_from_yield(y + h)
        minus = price_from_yield(y - h)
        up = (base - plus) / h * scale
        central = (minus - plus) / (2 * h) * scale
        rows.append({"bump_bp": bp, "one_sided": up, "centred": central,
                     "analytical": reference})
    return pd.DataFrame(rows)


def draw_convergence(ax) -> None:
    """Make small sensitivity differences visible as relative errors."""
    table = bump_comparison()
    for name, color, label in [("one_sided", BLUE, "One-sided"),
                                ("centred", ORANGE, "Centred")]:
        error = abs(table[name] - table.analytical) / table.analytical * 100
        ax.loglog(table.bump_bp, error, marker="o", color=color, label=label)
    ax.set(xlabel="Yield bump size (bp)", ylabel="Absolute relative error (%)")
    legend_above(ax)


def draw_values(ax) -> None:
    """Retain the book's one-bp comparison with meaningful precision."""
    result = finite_difference_dv01()
    bars = ax.bar(["1 bp upward price loss", "Analytical local DV01"],
                  [result["fd_dv01"], result["approx_dv01"]],
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.9f", padding=3, color=NAVY)
    ax.set(ylabel="Value per 100 principal", ylim=(0, 0.023))


def draw_nonlinearity(ax) -> None:
    """Compare a local linear loss with actual yield-bumped bond prices."""
    base_yield = 0.045
    base = price_from_yield(base_yield)
    dv01 = finite_difference_dv01()["approx_dv01"]
    shocks = np.array([-200, -100, -50, -25, 0, 25, 50, 100, 200])
    actual = np.array([base - price_from_yield(base_yield + bp * 1e-4)
                       for bp in shocks])
    gap = dv01 * shocks - actual
    ax.plot(shocks, gap, marker="o", color=ORANGE, lw=2)
    ax.axhline(0, color=NAVY, lw=0.7)
    ax.set(xlabel="Nominal semiannual-compounded yield shock (bp)",
           ylabel="Loss gap (per 100 face)")


def draw_inversion(ax) -> None:
    """Show the configured yield bracket and a target-price crossing."""
    yields = np.linspace(0, 0.20, 100)
    prices = [price_from_yield(y) for y in yields]
    recovered = yield_from_price(100)
    ax.plot(yields * 100, prices, color=BLUE, lw=2, label="Bond dirty price")
    ax.axhline(100, color=NAVY, ls="--", label="Target = 100")
    ax.axvline(recovered * 100, color=ORANGE, ls=":", lw=1.7)
    ax.scatter([recovered * 100], [100], color=ORANGE, s=45, zorder=4)
    ax.set(xlabel="Nominal annual yield, semiannual compounding (%)",
           ylabel="Dirty price per 100 face", xlim=(0, 20))
    legend_above(ax)


def main() -> None:
    """Export supplied fit diagnostics and analytical/bump comparisons."""
    export_panels("ch23_calibration_verification_diagnostics", [
        ("residuals", "Supplied quote residuals", draw_residuals),
        ("convergence", "Finite-difference convergence", draw_convergence),
        ("nonlinear", "Linear versus repriced loss", draw_nonlinearity),
        ("inversion", "Price/yield inversion", draw_inversion),
    ], book_panels=[
        ("residuals", "Quote residual comparison", draw_residuals),
        ("values", "One-bp sensitivity comparison", draw_values),
    ])


if __name__ == "__main__":
    main()
