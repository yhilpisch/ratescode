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
FEATURES = ["rate_level", "rate_slope", "spread", "volatility", "cs01"]


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


def logistic_scores() -> pd.DataFrame:
    """Compute interpretable logistic-style stress scores."""
    panel = load_panel()
    x = panel[FEATURES].to_numpy(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-12
    x_scaled = (x - mean) / scale
    weights = np.array([0.25, 0.15, 0.75, 0.65, 0.35])
    score = x_scaled @ weights
    panel["ml_score"] = 1.0 / (1.0 + np.exp(-score))
    panel["ml_alert"] = (panel["ml_score"] >= 0.55).astype(int)
    return panel[["date", "ml_score", "ml_alert", "stress"]]


def alert_metrics() -> pd.DataFrame:
    """Compare rule-based and ML-style alerts."""
    rule = rule_alerts()
    ml = logistic_scores()
    rows = []  # alert diagnostics
    for name, column, frame in [
        ("rule", "rule_alert", rule),
        ("ml", "ml_alert", ml),
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
    print(logistic_scores().round({"ml_score": 3}))
    print(alert_metrics().round(3))


if __name__ == "__main__":
    main()
