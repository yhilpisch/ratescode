"""
Python & AI for Rates, Bonds, and Credit
Chapter 30 · Credit Documents, Covenants, and Research Notes

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Covenant extraction controls figure for Chapter 30.
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

from ch30_credit_documents_covenants_research_notes import covenant_headroom


def draw_values(ax) -> None:
    """Identify maximum/minimum direction beside each ratio comparison."""
    table = covenant_headroom()
    y = np.arange(len(table))
    ax.hlines(y, table.extracted_value, table.threshold, color="#8CB6E8",
               lw=4)
    ax.scatter(table.extracted_value, y, color=BLUE, s=60, label="Observed")
    ax.scatter(table.threshold, y, color=ORANGE, s=100, marker="|",
               linewidths=2.5, label="Limit")
    ax.set(yticks=y, yticklabels=["Atlas: max leverage", "Beacon: min coverage",
                                 "Cedar: max leverage*"],
           xlabel="Covenant ratio (times)", xlim=(1.3, 5.1), ylim=(-0.5, 2.5))
    ax.invert_yaxis()
    legend_above(ax)


def draw_headroom(ax) -> None:
    """Calculate margins; do not substitute the broader stored review policy."""
    table = covenant_headroom()
    bars = ax.bar(["Atlas", "Beacon", "Cedar*"], table.headroom,
                  color=[ORANGE if h < 0.4 else BLUE for h in table.headroom])
    ax.bar_label(bars, fmt="%.2f", padding=3, color=NAVY)
    ax.axhline(0.4, color=NAVY, ls="--", label="Review below 0.40")
    ax.set(ylabel="Headroom (ratio points)", ylim=(0, 0.72),
           xlabel="*Cedar's covenant activation is not supplied")
    legend_above(ax, columns=1)


def main() -> None:
    """Export direction-labelled ratios and canonical computed headroom."""
    export_panels("ch30_covenant_extraction_controls", [
        ("values", "Metric-specific covenant direction", draw_values),
        ("headroom", "Conditional numerical headroom", draw_headroom),
    ])


if __name__ == "__main__":
    main()
