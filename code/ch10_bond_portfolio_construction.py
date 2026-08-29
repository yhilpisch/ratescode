"""
Python & AI for Rates, Bonds, and Credit
Chapter 10 · Bond Portfolio Construction and Risk Budgets

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


def portfolio_totals() -> dict[str, float]:
    """Aggregate market value, duration, and DV01 for the bond book."""
    table = pd.read_csv(DATA / "ch10_bond_portfolio.csv")
    value = float(table["market_value"].sum())  # total portfolio value
    weights = table["market_value"] / value  # market-value weights
    duration = float((weights * table["duration"]).sum())
    dv01 = float(table["dv01"].sum())
    return {"market_value": value, "duration": duration, "dv01": dv01}


def profile_exposures(total_value: float=10_000_000.0) -> pd.DataFrame:
    """Compute duration and DV01 for stylised allocation profiles."""
    bonds = pd.read_csv(DATA / "ch10_bond_portfolio.csv")
    profiles = pd.read_csv(DATA / "ch10_portfolio_profiles.csv")
    duration_map = dict(zip(bonds["bond"], bonds["duration"], strict=True))
    rows = []  # allocation diagnostics
    for _, row in profiles.iterrows():
        duration = sum(
            row[bond] * duration_map[bond] for bond in duration_map
        )  # weighted duration
        dv01 = total_value * duration / 10000.0  # value per bp
        rows.append({
            "portfolio": row["portfolio"],
            "duration": duration,
            "dv01": dv01,
        })
    return pd.DataFrame(rows)


def risk_budget() -> pd.DataFrame:
    """Return each bond's share of total portfolio DV01."""
    table = pd.read_csv(DATA / "ch10_bond_portfolio.csv")
    total_dv01 = float(table["dv01"].sum())  # aggregate DV01
    table["dv01_share"] = table["dv01"] / total_dv01
    return table[["bond", "dv01", "dv01_share"]]


def main() -> None:
    """Print a compact chapter result summary."""
    totals = portfolio_totals()  # aggregate risk measures
    print({key: round(value, 4) for key, value in totals.items()})
    print(profile_exposures().round(4))
    print(risk_budget().round(4))


if __name__ == "__main__":
    main()
