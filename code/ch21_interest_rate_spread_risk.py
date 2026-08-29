"""
Python & AI for Rates, Bonds, and Credit
Chapter 21 · Interest-Rate and Spread Risk

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
RATE_COLS = ["dv01_2y", "dv01_5y", "dv01_10y"]
SHOCK_COLS = ["rate_2y_bp", "rate_5y_bp", "rate_10y_bp"]


def load_portfolio() -> pd.DataFrame:
    """Load the frozen mixed rates and credit portfolio."""
    return pd.read_csv(DATA / "ch21_risk_portfolio.csv")


def exposure_summary() -> pd.DataFrame:
    """Aggregate rate and spread sensitivities by instrument type."""
    portfolio = load_portfolio()
    grouped = portfolio.groupby("type")[RATE_COLS + ["cs01"]].sum()
    return grouped.reset_index()


def scenario_pnl(hedged: bool=False) -> pd.DataFrame:
    """Compute sensitivity-based profit/loss under frozen shocks."""
    portfolio = load_portfolio()
    scenarios = pd.read_csv(DATA / "ch21_shock_scenarios.csv")
    exposures = portfolio[RATE_COLS + ["cs01"]].sum()  # base exposure
    if hedged:
        exposures["dv01_10y"] *= 0.40  # hedge 60 percent of 10Y risk
        exposures["cs01"] *= 0.50  # hedge half of spread risk
    rows = []  # scenario output rows
    for _, row in scenarios.iterrows():
        rate_loss = sum(
            exposures[dv] * row[shock]
            for dv, shock in zip(RATE_COLS, SHOCK_COLS, strict=True)
        )
        spread_loss = exposures["cs01"] * row["spread_bp"]
        rows.append({
            "scenario": row["scenario"],
            "pnl": -(rate_loss + spread_loss),
        })
    return pd.DataFrame(rows)


def hedge_comparison() -> pd.DataFrame:
    """Compare unhedged and hedged scenario profit/loss."""
    unhedged = scenario_pnl().rename(columns={"pnl": "unhedged_pnl"})
    hedged = scenario_pnl(hedged=True).rename(columns={"pnl": "hedged_pnl"})
    table = unhedged.merge(hedged, on="scenario")  # side-by-side P/L
    table["reduction"] = table["hedged_pnl"] - table["unhedged_pnl"]
    return table


def main() -> None:
    """Print a compact chapter result summary."""
    print(exposure_summary().round(2))
    print(hedge_comparison().round(2))


if __name__ == "__main__":
    main()
