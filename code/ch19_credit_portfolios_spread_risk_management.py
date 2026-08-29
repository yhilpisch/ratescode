"""
Python & AI for Rates, Bonds, and Credit
Chapter 19 · Credit Portfolios and Spread-Risk Management

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


def portfolio() -> pd.DataFrame:
    """Load the frozen synthetic credit portfolio."""
    table = pd.read_csv(DATA / "ch19_credit_portfolio.csv")
    table["cs01"] = table["market_value"] * table["spread_duration"] / 10000.0
    table["expected_loss"] = table["market_value"] * table["pd"]
    table["expected_loss"] *= 1.0 - table["recovery"]
    return table


def exposure_summary() -> pd.DataFrame:
    """Aggregate CS01 and market value by rating and sector."""
    table = portfolio()
    grouped = table.groupby("rating")[["market_value", "cs01"]].sum()
    return grouped.reset_index()


def stress_losses() -> pd.DataFrame:
    """Compute spread-widening losses by rating scenario."""
    table = portfolio()
    shocks = pd.read_csv(DATA / "ch19_spread_stress_scenarios.csv")
    rows = []  # scenario losses
    for _, shock in shocks.iterrows():
        loss = 0.0
        for _, row in table.iterrows():
            loss += row["cs01"] * shock[f"{row['rating']}_bp"]
        rows.append({"scenario": shock["scenario"], "loss": loss})
    return pd.DataFrame(rows)


def allocation_table() -> pd.DataFrame:
    """Load simple allocation candidates and rank by spread per CS01."""
    table = pd.read_csv(DATA / "ch19_allocation_candidates.csv")
    table["spread_per_cs01"] = table["spread_bp"] / table["cs01"]
    return table.sort_values("spread_per_cs01", ascending=False)


def main() -> None:
    """Print a compact chapter result summary."""
    print(exposure_summary().round(2))
    print(stress_losses().round(2))
    print(allocation_table().round(6))


if __name__ == "__main__":
    main()
