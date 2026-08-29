"""
Python & AI for Rates, Bonds, and Credit
Chapter 16 · Credit Spreads, Ratings, and Default Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def issuer_panel() -> pd.DataFrame:
    """Load the frozen synthetic issuer panel."""
    return pd.read_csv(DATA / "ch16_synthetic_issuer_panel.csv")


def expected_loss_table() -> pd.DataFrame:
    """Compute expected loss and spread pickup by issuer."""
    panel = issuer_panel()
    panel["lgd"] = 1.0 - panel["recovery"]  # loss given default
    panel["expected_loss_bp"] = panel["pd"] * panel["lgd"] * 10000.0
    panel["pickup_bp"] = panel["spread_bp"] - panel["expected_loss_bp"]
    return panel[["issuer", "rating", "spread_bp", "expected_loss_bp",
                  "pickup_bp"]]


def rating_summary() -> pd.DataFrame:
    """Aggregate spreads and expected loss by rating bucket."""
    table = expected_loss_table()
    grouped = table.groupby("rating")[["spread_bp", "expected_loss_bp"]]
    return grouped.mean().reset_index()


def spread_shock_loss(shock_bp: float=50.0) -> float:
    """Estimate portfolio loss from a parallel spread widening."""
    panel = issuer_panel()
    cs01 = panel["market_value"] * panel["spread_duration"] / 10000.0
    return float((cs01 * shock_bp).sum())


def main() -> None:
    """Print a compact chapter result summary."""
    print(expected_loss_table().round(2))
    print(rating_summary().round(2))
    print(f"spread_loss_50bp: {spread_shock_loss():.2f}")


if __name__ == "__main__":
    main()
