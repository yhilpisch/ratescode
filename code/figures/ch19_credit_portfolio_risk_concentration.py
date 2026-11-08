"""
Python & AI for Rates, Bonds, and Credit
Chapter 19 · Credit Portfolios and Spread-Risk Management

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit portfolio risk concentration figure for Chapter 19.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from credit_figure_layout import (
    NAVY, BLUE, ORANGE, export_panels, legend_above,
)

from ch19_credit_portfolios_spread_risk_management import portfolio


def draw_sectors(ax) -> None:
    """Draw market values in millions of reporting currency."""
    table = portfolio().groupby("sector").market_value.sum().sort_values()
    bars = ax.barh(table.index, table / 1e6, color=BLUE)
    ax.bar_label(bars, fmt="%.1f", padding=3, color=NAVY)
    ax.set(xlabel="Market value (million reporting currency)", xlim=(0, 7))


def draw_cs01(ax) -> None:
    """Draw rating sensitivities in an economically ordered sequence."""
    values = portfolio().groupby("rating").cs01.sum()
    values = values.reindex(["A", "BBB", "BB"])
    bars = ax.bar(values.index, values, color=BLUE)
    ax.bar_label(bars, fmt="%.0f", padding=3, color=NAVY)
    ax.set(xlabel="Rating bucket", ylabel="CS01 (reporting currency/bp)")
    ax.margins(y=0.18)


def draw_stress(ax) -> None:
    """Decompose the full five-name selloff into rating contributions."""
    cs01 = portfolio().groupby("rating").cs01.sum()
    values = cs01.reindex(["A", "BBB", "BB"]) * np.array([60, 100, 180])
    bars = ax.bar(values.index, values / 1000, color=ORANGE)
    ax.bar_label(bars, fmt="%.1f", padding=3, color=NAVY)
    ax.set(xlabel="Rating bucket; selloff widening = 60/100/180 bp",
           ylabel="Scenario loss (thousands)")
    ax.margins(y=0.2)


def main() -> None:
    """Export concentration, sensitivity, and stress contribution views."""
    export_panels("ch19_credit_portfolio_risk_concentration", [
        ("sectors", "Sector exposure", draw_sectors),
        ("cs01", "CS01 by rating", draw_cs01),
        ("stress", "Selloff loss contributions", draw_stress),
    ])


if __name__ == "__main__":
    main()
