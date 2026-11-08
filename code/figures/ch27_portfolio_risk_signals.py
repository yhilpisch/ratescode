"""
Python & AI for Rates, Bonds, and Credit
Chapter 27 · ML for Portfolio and Risk Signals

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Portfolio risk signal figure for Chapter 27.
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

from ch27_ml_portfolio_risk_signals import (
    load_panel, rule_alerts, illustrative_scores,
)


def date_axis(ax) -> None:
    """Keep four short date labels on the weekly synthetic panel."""
    table = load_panel()
    ticks = [0, 4, 8, 11]
    labels = pd.to_datetime(table.date).dt.strftime("%d %b")
    ax.set(xticks=ticks, xticklabels=labels.iloc[ticks],
           xlabel="Same-snapshot observation date (2026)")


def draw_loss(ax) -> None:
    """Mark rule alerts at the actual loss, including the false alert."""
    table = load_panel()
    x = np.arange(len(table))
    ax.plot(x, table.loss / 1000, color=BLUE, marker="o", ms=4,
            label="Supplied loss")
    stress = table.stress == 1
    ax.scatter(x[stress], table.loc[stress, "loss"] / 1000, s=90,
               facecolors="none", edgecolors=NAVY, marker="s",
               label="Stress label", zorder=3)
    alert = rule_alerts().rule_alert == 1
    ax.scatter(x[alert], table.loc[alert, "loss"] / 1000, color=ORANGE,
               marker="v", s=45, label="Rule alert", zorder=4)
    ax.set_ylabel("Supplied loss ($ thousands)")
    date_axis(ax)
    legend_above(ax, columns=3)


def draw_scores(ax) -> None:
    """Show every score, including the two missed stress observations."""
    table = illustrative_scores()
    x = np.arange(len(table))
    ax.plot(x, table.score, color=ORANGE, marker="o", label="Fixed score")
    ax.axhline(0.55, color=NAVY, ls="--", label="Threshold = 0.55")
    missed = (table.stress == 1) & (table.score_alert == 0)
    ax.scatter(x[missed], table.loc[missed, "score"], s=95,
               facecolors="none", edgecolors=NAVY, marker="s",
               label="Missed stress", zorder=4)
    ax.set(ylabel="Uncalibrated fixed score", ylim=(0, 1.05))
    date_axis(ax)
    legend_above(ax, columns=3)


def draw_decisions(ax) -> None:
    """Align all labels and both rules, rather than selecting alert dates."""
    from matplotlib.colors import ListedColormap
    panel = load_panel()
    values = np.vstack([panel.stress, rule_alerts().rule_alert,
                        illustrative_scores().score_alert])
    ax.imshow(values, cmap=ListedColormap(["#F1F5FA", BLUE]),
              vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for row in range(3):
        for col in range(values.shape[1]):
            ax.text(col, row, str(values[row, col]), ha="center",
                    va="center", color="white" if values[row, col] else NAVY,
                    fontsize=10)
    ax.set_yticks(range(3),
                  ["Supplied stress", "Threshold rule", "Fixed score"])
    date_axis(ax)


def main() -> None:
    """Export full-panel loss, score, and aligned-decision diagnostics."""
    export_panels("ch27_portfolio_risk_signals", [
        ("loss", "Full loss path and rule alerts", draw_loss),
        ("scores", "All fixed teaching scores", draw_scores),
        ("decisions", "Full-panel alert decisions", draw_decisions),
    ])


if __name__ == "__main__":
    main()
