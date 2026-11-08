"""
Python & AI for Rates, Bonds, and Credit
Chapter 21 · Interest-Rate and Spread Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Rate and spread risk map figure for Chapter 21.
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

from ch21_interest_rate_spread_risk import load_portfolio, hedge_comparison


def draw_exposures(ax) -> None:
    """Show all instrument sensitivities with explicit currency/bp units."""
    table = load_portfolio()
    values = table[["dv01_2y", "dv01_5y", "dv01_10y", "cs01"]].to_numpy()
    im = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(4), ["2Y DV01", "5Y DV01", "10Y DV01", "CS01"])
    ax.set_yticks(range(len(table)), table.instrument)
    for row in range(len(table)):
        for col in range(4):
            color = "white" if values[row, col] > 400 else NAVY
            ax.text(col, row, str(values[row, col]), ha="center",
                    va="center", color=color, fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Dollars/bp")


def draw_scenarios(ax) -> None:
    """Compare positive losses, rather than the canonical signed P/L."""
    table = hedge_comparison()
    y = np.arange(len(table))
    for column, shift, color, label in [
        ("unhedged_pnl", -0.18, BLUE, "Unhedged"),
        ("hedged_pnl", 0.18, ORANGE, "Idealised overlay"),
    ]:
        ax.barh(y + shift, -table[column] / 1000, height=0.32,
                color=color, label=label)
    ax.set_yticks(y, table.scenario)
    ax.invert_yaxis()
    ax.set_xlabel("Scenario loss ($ thousands)")
    legend_above(ax)


def draw_contributions(ax) -> None:
    """Decompose the hypothetical recession shock before/after overlay."""
    exposures = load_portfolio()[
        ["dv01_2y", "dv01_5y", "dv01_10y", "cs01"]
    ].sum().to_numpy(float)
    shock = np.array([-20, 15, 45, 80])
    residual = exposures * np.array([1, 1, 0.4, 0.5])
    x = np.arange(4)
    ax.bar(x - 0.18, exposures * shock / 1000, width=0.36,
           color=BLUE, label="Unhedged")
    ax.bar(x + 0.18, residual * shock / 1000, width=0.36,
           color=ORANGE, label="Idealised overlay")
    ax.axhline(0, color=NAVY, lw=1)
    ax.set_xticks(x, ["2Y rate", "5Y rate", "10Y rate", "Credit spread"])
    ax.set_ylabel("Loss contribution ($ thousands)")
    legend_above(ax)


def historical_move() -> pd.Series:
    """Select the largest absolute 10Y daily move in the frozen 2024 panel."""
    path = ROOT / "data/ch21_us_rates_inflation_shock_history_official.csv"
    history = pd.read_csv(path)
    return history.loc[history.delta_10y_bp.abs().idxmax()]


def draw_historical(ax) -> None:
    """Decompose the observed nominal move into real and breakeven parts."""
    row = historical_move()
    values = [row.delta_10y_bp, row.delta_real_10y_bp,
              row.delta_breakeven_10y_bp]
    bars = ax.bar(["Nominal 10Y", "Real 10Y", "Breakeven 10Y"], values,
                  color=[NAVY, BLUE, ORANGE])
    ax.bar_label(bars, fmt="%+.0f bp", padding=3, color=NAVY)
    ax.set(ylabel="One-day change (bp)",
           xlabel=f"Treasury snapshot: {row.date}; nominal = real + breakeven")
    ax.margins(y=0.2)


def main() -> None:
    """Create the book composite and four readable slide views."""
    export_panels("ch21_rate_spread_risk_map", [
        ("exposures", "Instrument sensitivities", draw_exposures),
        ("scenarios", "Named scenario losses", draw_scenarios),
        ("contributions", "Recession-shock contributions", draw_contributions),
        ("historical", "Observed yield-move decomposition", draw_historical),
    ])


if __name__ == "__main__":
    main()
