"""
Python & AI for Rates, Bonds, and Credit
Chapter 26 · ML for Credit Spreads and Default Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Credit ML diagnostics figure for Chapter 26.
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

from ch26_ml_credit_spreads_default_risk import (
    classification_table, chronological_split,
)


def draw_holdout(ax) -> None:
    """Display the three actual held-out scores and the fixed threshold."""
    table = classification_table()
    bars = ax.bar(table.issuer, table.probability,
                  color=[BLUE if y == 0 else ORANGE for y in table.label])
    ax.bar_label(bars, fmt="%.3f", padding=3, color=NAVY)
    ax.axhline(0.5, color=NAVY, ls="--", label="Threshold = 0.50")
    ax.set(ylabel="Uncalibrated fitted score", ylim=(0, 1.15),
           xlabel="23 January 2026; true labels: 0, 1, 1")
    legend_above(ax, columns=1)


def draw_association(ax) -> None:
    """Use training snapshots only; correlation is not model importance."""
    train, _ = chronological_split()
    columns = ["spread_mom", "leverage", "coverage", "macro"]
    corr = train[columns].corrwith(train.label).abs().sort_values()
    ax.barh(corr.index, corr, color=ORANGE)
    ax.set(xlabel="Absolute training correlation (9 rows)", xlim=(0, 1),
           yticks=range(len(corr)),
           yticklabels=[name.replace("spread_mom", "Momentum")
                        .capitalize() for name in corr.index])


def draw_hand_scores(ax) -> None:
    """Make the missed Beta case visible in the separate four-row rule."""
    momentum = np.array([2, 13, 25, 20])
    leverage = np.array([3.2, 4.8, 6.1, 5.2])
    coverage = np.array([4.5, 2.7, 1.6, 2.2])
    index = -6 + 0.15 * momentum + leverage - 0.6 * coverage
    score = 1 / (1 + np.exp(-index))
    y = np.arange(4)
    ax.scatter(score, y, color=[BLUE, ORANGE, ORANGE, ORANGE], s=65)
    ax.axvline(0.5, color=NAVY, ls="--", label="Threshold = 0.50")
    for value, pos in zip(score, y):
        ax.annotate(f"{value:.3f}", (value, pos), xytext=(5, 6),
                    textcoords="offset points", color=NAVY, fontsize=10)
    ax.set(yticks=y, yticklabels=["Alpha: 0", "Beta: 1", "Gamma: 1",
                                 "Delta: 1"], xlim=(-0.05, 1.1),
           ylim=(-0.5, 3.5), xlabel="Hand-specified sigmoid score")
    ax.invert_yaxis()
    legend_above(ax, columns=1)


def draw_supplied_recall(ax) -> None:
    """Retain the book's explicitly hand-constructed recall comparison."""
    table = pd.read_csv(DATA / "ch26_credit_classification_metrics.csv")
    bars = ax.bar(table.model, table.recall, color=BLUE)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=NAVY)
    ax.set(ylabel="Supplied illustrative recall", ylim=(0, 1.05))


def draw_full_association(ax) -> None:
    """Retain the book's descriptive association on the complete panel."""
    panel = pd.read_csv(DATA / "ch26_credit_ml_panel.csv")
    columns = ["spread_mom", "leverage", "coverage", "macro"]
    corr = panel[columns].corrwith(panel.label).abs().sort_values()
    ax.barh(corr.index, corr, color=ORANGE)
    ax.set(xlabel="Absolute full-panel correlation", xlim=(0, 1))


def main() -> None:
    """Keep supplied book illustrations separate from fitted slide data."""
    export_panels("ch26_credit_ml_diagnostics", [
        ("holdout", "Fitted later-snapshot scores", draw_holdout),
        ("association", "Training feature associations", draw_association),
        ("hand", "Separate hand-specified scores", draw_hand_scores),
    ], book_panels=[
        ("supplied", "Separate supplied recall illustration",
         draw_supplied_recall),
        ("full", "Descriptive feature association", draw_full_association),
    ])


if __name__ == "__main__":
    main()
