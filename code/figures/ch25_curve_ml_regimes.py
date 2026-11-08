"""
Python & AI for Rates, Bonds, and Credit
Chapter 25 · ML for Yield Curves and Rates

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Curve ML regime diagnostics figure for Chapter 25.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "code"))
from credit_figure_layout import (
    NAVY, BLUE, ORANGE, export_panels, legend_above,
)

from ch25_ml_yield_curves_rates import ridge_forecast


def draw_regimes(ax) -> None:
    """Show all twelve descriptive states with the stated slope cutoffs."""
    table = pd.read_csv(DATA / "ch25_curve_regimes.csv")
    for name, color in [("calm", BLUE), ("transition", "#8CB6E8"),
                         ("stress", ORANGE)]:
        rows = table.loc[table.regime == name]
        ax.scatter(rows.slope, rows.level, c=color, s=50, label=name)
    for cutoff in [0.9, 1.2]:
        ax.axvline(cutoff, color=NAVY, ls=":", lw=1)
    ax.set(xlabel="10Y - 1Y slope (pp)", ylabel="Four-node level (%)")
    legend_above(ax, columns=3)


def draw_errors(ax) -> None:
    """Compare magnitude errors in basis points on the four test rows."""
    table = ridge_forecast()
    errors = [(table.target - table[column]).abs().mean() * 100
              for column in ["random_walk", "ridge"]]
    bars = ax.bar(["Zero change", "Fitted ridge"], errors,
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.2f", padding=3, color=NAVY)
    ax.set(ylabel="Holdout MAE (bp)", ylim=(0, 27))


def draw_forecast(ax) -> None:
    """Expose the sign error that an average error metric conceals."""
    table = ridge_forecast()
    x = np.arange(len(table))
    ax.plot(x, table.target * 100, marker="o", color=NAVY,
            label="Observed 10Y change")
    ax.plot(x, table.ridge * 100, marker="o", color=ORANGE,
            label="Ridge forecast")
    ax.axhline(0, color=BLUE, ls="--", label="Zero-change forecast")
    ax.set(xticks=x, xticklabels=table.date.dt.strftime("%d %b"),
           xlabel="Feature date (2026)", ylabel="Next weekly 10Y change (bp)")
    legend_above(ax, columns=3)


def draw_official(ax) -> None:
    """Show real shape features without implying a trained market model."""
    table = pd.read_csv(DATA / "ch25_us_treasury_curve_ml_panel_official.csv")
    dates = pd.to_datetime(table.date)
    ax.plot(dates, table.slope_10y_2y_bp, color=BLUE,
            label="10Y - 2Y slope")
    ax.plot(dates, table.curvature_5y_bp, color=ORANGE,
            label="2 x 5Y - 2Y - 10Y")
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set(xlabel="Treasury snapshot date (2024)",
           ylabel="Par-yield shape (bp)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    legend_above(ax)


def main() -> None:
    """Export synthetic regime/forecast views and sourced curve features."""
    export_panels("ch25_curve_ml_regimes", [
        ("regimes", "Descriptive slope-rule regimes", draw_regimes),
        ("errors", "Four-row forecast errors", draw_errors),
        ("forecast", "Forecasts by date", draw_forecast),
        ("official", "Observed Treasury shape features", draw_official),
    ])


if __name__ == "__main__":
    main()
