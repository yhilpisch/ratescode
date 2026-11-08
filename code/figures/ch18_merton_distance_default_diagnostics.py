"""
Python & AI for Rates, Bonds, and Credit
Chapter 18 · Structural Credit Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Merton distance-to-default diagnostics figure for Chapter 18.
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

from ch18_structural_credit_models import scenario_table


def draw_distance(ax) -> None:
    """Draw canonical risk-neutral distances for all four scenarios."""
    table = scenario_table()
    bars = ax.barh(table.scenario, table.distance_to_default, color=BLUE)
    ax.bar_label(bars, fmt="%.3f", padding=4, color=NAVY)
    ax.axvline(0, color=NAVY, lw=1)
    ax.invert_yaxis()
    ax.set(xlabel="Risk-neutral distance to default", xlim=(-0.25, 0.95))


def draw_probability(ax) -> None:
    """Draw terminal PDs rather than historical frequencies."""
    table = scenario_table()
    bars = ax.barh(table.scenario, table.default_probability * 100,
                   color=ORANGE)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, color=NAVY)
    ax.invert_yaxis()
    ax.set(xlabel="One-year risk-neutral default probability (%)",
           xlim=(0, 65))


def draw_payoffs(ax) -> None:
    """Allocate terminal assets to debt and equity with face value 100."""
    assets = np.linspace(0, 200, 201)
    ax.axvspan(0, 100, color=BLUE, alpha=0.08)
    ax.plot(assets, np.minimum(assets, 100), color=BLUE, lw=2,
            label="Debt: min(assets, 100)")
    ax.plot(assets, np.maximum(assets - 100, 0), color=ORANGE, lw=2,
            label="Equity: max(assets - 100, 0)")
    ax.axvline(100, color=NAVY, ls="--", lw=0.8)
    ax.set(xlabel="Asset value at debt maturity (per 100 face)",
           ylabel="Terminal claim payoff", xlim=(0, 200), ylim=(-3, 110))
    legend_above(ax)


def main() -> None:
    """Create model diagnostics and the terminal payoff decomposition."""
    export_panels("ch18_merton_distance_default_diagnostics", [
        ("distance", "Distance to default", draw_distance),
        ("probability", "Terminal model default probability", draw_probability),
        ("payoffs", "Debt and equity at maturity", draw_payoffs),
    ])


if __name__ == "__main__":
    main()
