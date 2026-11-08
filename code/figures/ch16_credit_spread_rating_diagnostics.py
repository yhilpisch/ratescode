"""
Python & AI for Rates, Bonds, and Credit
Chapter 16 · Credit Spreads, Ratings, and Default Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit spread and rating diagnostics figure for Chapter 16.
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

from ch16_credit_spreads_ratings_default_risk import expected_loss_table


def draw_ratings(ax) -> None:
    """Draw unweighted rating averages from all five synthetic issuers."""
    values = expected_loss_table().groupby("rating").spread_bp.mean()
    values = values.reindex(["A", "BBB", "BB"])
    bars = ax.bar(values.index, values, color=BLUE)
    ax.bar_label(bars, fmt="%.1f", padding=3, color=NAVY)
    ax.set(xlabel="Rating bucket", ylabel="Annual spread quote (bp)")
    ax.margins(y=0.18)


def draw_losses(ax) -> None:
    """Compare distinct measures with their horizons in the legend."""
    table = expected_loss_table()
    x = np.arange(len(table))
    ax.bar(x - 0.18, table.spread_bp, width=0.36, color=BLUE,
           label="Spread (annual bp)")
    ax.bar(x + 0.18, table.expected_loss_bp, width=0.36, color=ORANGE,
           label="One-year EL (bp of EAD)")
    ax.set_xticks(x, [name.split()[0] for name in table.issuer])
    ax.set(xlabel="Synthetic issuer", ylabel="Basis points")
    legend_above(ax)


def main() -> None:
    """Create rating and loss comparisons without overlapping bars."""
    export_panels("ch16_credit_spread_rating_diagnostics", [
        ("ratings", "Average spread by rating", draw_ratings),
        ("loss", "Spread and one-year loss fraction", draw_losses),
    ])


if __name__ == "__main__":
    main()
