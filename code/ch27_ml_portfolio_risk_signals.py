"""
Python & AI for Rates, Bonds, and Credit
Chapter 27 · ML for Portfolio and Risk Signals

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
def load_panel() -> pd.DataFrame:
    """Load the frozen portfolio signal panel."""
    return pd.read_csv(DATA / "ch27_portfolio_signal_panel.csv")


def rule_alerts() -> pd.DataFrame:
    """Compute a transparent rule-based stress alert."""
    panel = load_panel()
    panel["rule_alert"] = (
        (panel["spread"] > 150) | (panel["volatility"] > 18)
    ).astype(int)
    return panel[["date", "rule_alert", "stress"]]


def illustrative_scores() -> pd.DataFrame:
    """Apply the chapter's fixed, hand-specified stress-score formula.

    This is a transparent teaching rule, not a fitted ML model. Fixed
    coefficients avoid estimating normalization from future panel rows.
    """
    panel = load_panel()
    score = -8.0 + 0.03 * panel["spread"] + 0.18 * panel["volatility"]
    panel["score"] = 1.0 / (1.0 + np.exp(-score))
    panel["score_alert"] = (panel["score"] >= 0.55).astype(int)
    return panel[["date", "score", "score_alert", "stress"]]


def alert_metrics() -> pd.DataFrame:
    """Compare alerts descriptively on the same synthetic sample."""
    rule = rule_alerts()
    score = illustrative_scores()
    rows = []  # alert diagnostics
    for name, column, frame in [
        ("rule", "rule_alert", rule),
        ("fixed score", "score_alert", score),
    ]:
        tp = int(((frame[column] == 1) & (frame["stress"] == 1)).sum())
        fp = int(((frame[column] == 1) & (frame["stress"] == 0)).sum())
        fn = int(((frame[column] == 0) & (frame["stress"] == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"model": name, "precision": precision, "recall": recall})
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(illustrative_scores().round({"score": 3}))
    print(alert_metrics().round(3))


if __name__ == "__main__":
    main()
