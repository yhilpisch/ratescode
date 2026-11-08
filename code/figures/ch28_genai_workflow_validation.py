"""
Python & AI for Rates, Bonds, and Credit
Chapter 28 · GenAI Workflows for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

GenAI extraction and validation figure for Chapter 28.
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

from ch28_genai_workflows_fixed_income import load_extractions
from genai_figure_checks import extraction_checks


def draw_confidence(ax) -> None:
    """Show queue selection rather than implying extraction accuracy."""
    table = load_extractions()
    queued = (table.confidence < 0.85) | table.needs_review.eq(1)
    labels = ["DV01", "10Y yield change", "Spread change", "Cash buffer",
              "Weekly outflow"]
    bars = ax.barh(labels, table.confidence,
                   color=[ORANGE if q else BLUE for q in queued])
    ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")
    ax.axvline(0.85, color=NAVY, ls="--", label="Queue cutoff = 0.85")
    ax.set(xlabel="Supplied confidence (uncalibrated)", xlim=(0, 1.08))
    ax.invert_yaxis()
    legend_above(ax, columns=1)


def draw_checks(ax) -> None:
    """Count recomputed agreement on the five supplied comparisons."""
    counts = extraction_checks().computed_status.value_counts()
    counts = counts.reindex(["pass", "fail"], fill_value=0)
    bars = ax.bar(["Numeric/date agreement", "Mismatch"], counts,
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.0f", padding=3, color=NAVY)
    ax.set(ylabel="Supplied comparisons", ylim=(0, 5), yticks=range(6))


def main() -> None:
    """Export separate confidence and recomputed-check views."""
    export_panels("ch28_genai_workflow_validation", [
        ("confidence", "Supplied confidence and review queue", draw_confidence),
        ("checks", "Recomputed draft comparisons", draw_checks),
    ])


if __name__ == "__main__":
    main()
