"""
Python & AI for Rates, Bonds, and Credit
Chapter 32 · Automated Reporting and Communication

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Report reconciliation figure for Chapter 32.
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

from genai_figure_checks import report_checks


def draw_relative(ax) -> None:
    """Normalise discrepancy before plotting metrics with unlike units."""
    table = report_checks()
    if table.source_value.eq(0).any():
        raise ValueError("relative errors require a nonzero reference")
    error = table.difference.abs() / table.source_value.abs() * 100
    labels = ["Market value", "Duration", "DV01", "Weekly P/L", "Stress P/L"]
    colors = [ORANGE if value > 0 else BLUE for value in error]
    bars = ax.barh(labels, error, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, color=NAVY)
    ax.invert_yaxis()
    ax.set(xlabel="Absolute discrepancy / absolute source (%)", xlim=(0, 7.7))


def draw_weekly(ax) -> None:
    """Compare values in the same unit for the failed weekly P/L claim."""
    row = report_checks().set_index("metric").loc["weekly_pnl"]
    bars = ax.bar(["Source", "Draft"], [row.source_value, row.draft_value],
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.1f", padding=4, color=NAVY)
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set(ylabel="Weekly P/L (USD million)", ylim=(-4, 0.5))


def main() -> None:
    """Export comparable relative errors and the unit-specific failed claim."""
    export_panels("ch32_report_reconciliation", [
        ("relative", "Unit-normalised draft discrepancies", draw_relative),
        ("weekly", "Understated weekly loss", draw_weekly),
    ])


if __name__ == "__main__":
    main()
