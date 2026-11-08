"""
Python & AI for Rates, Bonds, and Credit
Chapter 20 · Private Credit Analytics

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Private credit stress dashboard figure for Chapter 20.
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

from ch20_private_credit_analytics import (
    loan_book, borrower_stress, stress_table,
)


def draw_headroom(ax, field: str, label: str) -> None:
    """Compare base and severe headroom against individual covenants."""
    details = borrower_stress()
    names = loan_book().borrower.tolist()
    y = np.arange(len(names))
    for scenario, shift, color in [("base", -0.18, BLUE),
                                    ("severe stress", 0.18, ORANGE)]:
        table = details.loc[details.scenario == scenario].set_index("borrower")
        values = table.loc[names, field]
        bars = ax.barh(y + shift, values, height=0.32, color=color,
                       label="Base" if scenario == "base" else "Severe stress")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, color=NAVY)
    ax.set_yticks(y, [name.split()[0] for name in names])
    ax.axvline(0, color=NAVY, lw=1)
    ax.invert_yaxis()
    ax.margins(x=0.25)
    ax.set_xlabel(label)
    legend_above(ax)


def draw_leverage(ax) -> None:
    """Draw leverage headroom, retaining the base Cobalt breach."""
    draw_headroom(ax, "leverage_headroom", "Leverage headroom (turns)")


def draw_coverage(ax) -> None:
    """Use borrower-specific coverage limits rather than a common cutoff."""
    draw_headroom(ax, "coverage_headroom", "Coverage headroom (turns)")


def draw_book_leverage(ax) -> None:
    """Retain the book's base leverage view with explicit ratio units."""
    book = loan_book()
    ax.barh([name.split()[0] for name in book.borrower],
            book.debt / book.ebitda, color=BLUE)
    ax.set_xlabel("Debt/annual EBITDA (turns)")


def draw_book_minimum(ax) -> None:
    """Retain canonical minima without treating them as covenant tests."""
    table = stress_table()
    ax.barh(table.scenario, table.min_coverage, color=ORANGE)
    ax.set_xlabel("Minimum borrower coverage (turns)")


def main() -> None:
    """Keep the book view and add complete borrower headroom slide panels."""
    export_panels("ch20_private_credit_stress_dashboard", [
        ("leverage", "Leverage headroom", draw_leverage),
        ("coverage", "Coverage headroom", draw_coverage),
    ], book_panels=[
        ("base", "Base borrower leverage", draw_book_leverage),
        ("minimum", "Minimum coverage by scenario", draw_book_minimum),
    ])


if __name__ == "__main__":
    main()
