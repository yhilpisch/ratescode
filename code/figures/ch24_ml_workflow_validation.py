"""
Python & AI for Rates, Bonds, and Credit
Chapter 24 · Machine Learning Workflow for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Machine learning workflow and validation figure for Chapter 24.
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

from ch24_machine_learning_workflow_fixed_income import (
    load_panel, model_comparison, ridge_predict, time_split,
)


def draw_split(ax) -> None:
    """Show the actual seven-row fit, one-row purge, and four-row test."""
    train, test = time_split()
    purged = len(load_panel()) - len(train) - len(test)
    sizes = [len(train), purged, len(test)]
    left = 0
    for size, color, label in zip(sizes, [BLUE, "#8CB6E8", ORANGE],
                                  ["Training", "Purged", "Holdout"]):
        ax.barh([0], size, left=left, color=color, label=label, height=0.5)
        ax.text(left + size / 2, 0, str(size), ha="center", va="center",
                color=NAVY if label == "Purged" else "white", fontsize=13)
        left += size
    ax.set(xlim=(0, sum(sizes)), ylim=(-0.8, 0.8), yticks=[],
           xlabel="Ordered observation count", xticks=[0, 4, 8, 12])
    legend_above(ax, columns=3)


def draw_errors(ax) -> None:
    """Use fitted canonical metrics, not the separate supplied snapshot."""
    table = model_comparison()
    bars = ax.bar(["Training mean", "Fitted ridge"], table.mae * 100,
                  color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.2f", padding=3, color=NAVY)
    ax.set(ylabel="Holdout MAE (bp)", ylim=(0, 14))


def draw_predictions(ax) -> None:
    """Compare forecasts with the same four observed targets."""
    train, _ = time_split()
    table = ridge_predict()
    x = np.arange(len(table))
    ax.plot(x, table.target * 100, marker="o", color=NAVY,
            label="Observed change")
    ax.plot(x, table.prediction * 100, marker="o", color=ORANGE,
            label="Ridge forecast")
    ax.axhline(train.target.mean() * 100, color=BLUE, ls="--",
               label="Training mean")
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set(xticks=x, xticklabels=table.date.dt.strftime("%d %b"),
           xlabel="Feature date (2026)", ylabel="Next-interval change (bp)")
    legend_above(ax, columns=3)


def draw_supplied_errors(ax) -> None:
    """Preserve the book's explicitly separate illustrative metric table."""
    table = pd.read_csv(DATA / "ch24_model_comparison.csv")
    bars = ax.bar(table.model, table.mae, color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.4f", padding=3, color=NAVY)
    ax.set(ylabel="Supplied MAE (pp)", ylim=(0, 0.14))


def main() -> None:
    """Export the book illustration and three fitted-workflow slide views."""
    export_panels("ch24_ml_workflow_validation", [
        ("split", "Purged chronological split", draw_split),
        ("errors", "Fitted holdout errors", draw_errors),
        ("predictions", "Four holdout forecasts", draw_predictions),
    ], book_panels=[
        ("split", "Purged chronological split", draw_split),
        ("supplied", "Separate illustrative metric snapshot",
         draw_supplied_errors),
    ])


if __name__ == "__main__":
    main()
