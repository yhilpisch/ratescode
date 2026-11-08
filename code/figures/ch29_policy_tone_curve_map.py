"""
Python & AI for Rates, Bonds, and Credit
Chapter 29 · Central-Bank and Macro Document Intelligence

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Policy tone and curve-reaction figure for Chapter 29.
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

from genai_figure_checks import policy_events


def draw_tone(ax) -> None:
    """Separate stance and concern scales, with explicit illustrative labels."""
    table = policy_events()
    x = np.arange(len(table))
    ax.plot(x, table.policy_stance, marker="o", color=BLUE, label="Stance")
    ax.plot(x, table.inflation_concern, marker="s", color=ORANGE,
            label="Inflation concern")
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set(xticks=x, xticklabels=["A: 18 Mar", "B: 29 Apr", "C: 10 Jun"],
           ylabel="Supplied score", xlabel="Synthetic event date (2026)",
           ylim=(-1, 1.1))
    legend_above(ax)


def draw_moves(ax) -> None:
    """Show the two yield moves and their arithmetic slope difference."""
    table = policy_events()
    x = np.arange(len(table))
    for offset, col, color, label in [
        (-0.25, "two_year_change", BLUE, "2Y change"),
        (0, "ten_year_change", ORANGE, "10Y change"),
        (0.25, "curve_slope_change", NAVY, "10Y - 2Y change"),
    ]:
        bars = ax.bar(x + offset, table[col], width=0.24,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%+.0f", padding=2, color=NAVY, fontsize=10)
    ax.axhline(0, color=NAVY, lw=0.7)
    ax.set(xticks=x, xticklabels=["A: 18 Mar", "B: 29 Apr", "C: 10 Jun"],
           ylabel="Yield/slope change (bp)",
           xlabel="Synthetic event date (2026)", ylim=(-7, 14))
    legend_above(ax, columns=3)


def draw_association(ax) -> None:
    """Retain the book's association view without an estimated slope."""
    table = policy_events()
    ax.scatter(table.policy_stance, table.two_year_change,
               color=BLUE, s=70)
    for row in table.itertuples():
        ax.annotate(row.statement_id.replace("meeting_", "").upper(),
                    (row.policy_stance, row.two_year_change),
                    xytext=(5, 5), textcoords="offset points", color=NAVY)
    ax.set(xlabel="Supplied stance", ylabel="2Y change (bp)",
           xlim=(-0.3, 1.1), ylim=(-8, 14))


def draw_official(ax) -> None:
    """Show all fifteen available daily release windows, sorted by release."""
    table = pd.read_csv(DATA / "ch29_fomc_curve_event_panel_official.csv")
    table = table.sort_values("release_date").reset_index(drop=True)
    x = np.arange(len(table))
    ax.bar(x - 0.18, table.delta_2y_bp, width=0.36, color=BLUE, label="2Y")
    ax.bar(x + 0.18, table.delta_10y_bp, width=0.36, color=ORANGE, label="10Y")
    ax.axhline(0, color=NAVY, lw=0.7)
    labels = pd.to_datetime(table.release_date).dt.strftime("%d %b")
    ax.set(xticks=x, xticklabels=labels, ylabel="Daily par-yield change (bp)",
           xlabel="2024 release date; S = statement, M = minutes")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    for index, row in table.iterrows():
        ax.text(index, 26, "S" if row.doc_type == "statement" else "M",
                ha="center", color=NAVY, fontsize=9)
    ax.set_ylim(-14, 29)
    legend_above(ax)


def main() -> None:
    """Export toy tone/moves and the separate sourced daily event windows."""
    export_panels("ch29_policy_tone_curve_map", [
        ("tone", "Synthetic extracted tone", draw_tone),
        ("moves", "Synthetic yield and slope changes", draw_moves),
        ("official", "Observed daily release windows", draw_official),
    ], book_panels=[
        ("tone", "Synthetic extracted tone", draw_tone),
        ("association", "Three-event association illustration",
         draw_association),
    ])


if __name__ == "__main__":
    main()
