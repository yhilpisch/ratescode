"""
Python & AI for Rates, Bonds, and Credit
Chapter 31 · Tool-Using Agents for Fixed-Income Analysis

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Agent tool-validation figure for Chapter 31.
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

from genai_figure_checks import agent_checks


def draw_claims(ax) -> None:
    """Keep unlike units in labelled rows instead of a common value axis."""
    table = agent_checks()
    labels = ["10Y rate (%)", "Clean price (quote)", "Stress P/L (USD m)",
              "Initial failure count"]
    cells = [[label, f"{row.tool_value:g}", f"{row.draft_value:g}",
              row.computed_status] for label, row in
             zip(labels, table.itertuples())]
    ax.axis("off")
    artist = ax.table(cellText=cells,
                      colLabels=["Within-log claim", "Reference", "Draft",
                                 "Numeric check"],
                      colWidths=[0.44, 0.17, 0.17, 0.22],
                      cellLoc="left", colLoc="left", loc="center")
    artist.auto_set_font_size(False)
    artist.set_fontsize(12 if ax.figure.get_figwidth() < 10 else 10)
    artist.scale(1, 1.65)
    for (row, col), cell in artist.get_celld().items():
        cell.set_edgecolor("#D8E1EB")
        cell.get_text().set_color(NAVY)
        if row == 0:
            cell.set_facecolor("#E7EEF8")
        elif row >= 3:
            cell.set_facecolor("#FFF0E0")


def draw_counts(ax) -> None:
    """Count executed agreement tests, including the false report claim."""
    table = agent_checks()
    counts = table.computed_status.value_counts().reindex(["pass", "fail"])
    bars = ax.bar(["Numeric agreement", "Mismatch"], counts,
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.0f", padding=3, color=NAVY)
    ax.set(ylabel="Within-log comparisons", ylim=(0, 3), yticks=range(4))


def main() -> None:
    """Export unit-labelled claim rows and their narrow numeric-check counts."""
    export_panels("ch31_agent_tool_validation_flow", [
        ("claims", "Within-log claim comparisons", draw_claims),
        ("counts", "Recomputed numeric agreement", draw_counts),
    ])


if __name__ == "__main__":
    main()
