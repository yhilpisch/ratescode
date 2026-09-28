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


def allocation_table(notional: float=10_000_000.0) -> pd.DataFrame:
    """Compare candidate carry and losses on the same synthetic notional.

    Candidate CS01 inputs are assumed to refer to this 10m portfolio.
    Default rates are treated as one-year probabilities; rating/recovery
    inputs are issuer-weighted proxies. The stress assumes candidate CS01
    follows the candidate rating weights.
    """
    if notional != 10_000_000.0:
        raise ValueError("candidate CS01 values assume a 10m notional")
    table = pd.read_csv(DATA / "ch19_allocation_candidates.csv")
    issuers = portfolio()
    loss_rate = issuers["pd"] * (1.0 - issuers["recovery"])
    proxies = (
        (issuers["market_value"] * loss_rate).groupby(issuers["rating"]).sum()
        / issuers.groupby("rating")["market_value"].sum()
    )
    shock = pd.read_csv(DATA / "ch19_spread_stress_scenarios.csv")
    selloff = shock.set_index("scenario").loc["credit selloff"]
    weights = [f"{rating}_weight" for rating in ["A", "BBB", "BB"]]
    table["annual_spread_carry"] = notional * table["spread_bp"] / 10000.0
    table["expected_loss_proxy"] = notional * sum(
        table[f"{rating}_weight"] * proxies[rating]
        for rating in ["A", "BBB", "BB"]
    )
    table["selloff_loss"] = table["cs01"] * sum(
        table[f"{rating}_weight"] * selloff[f"{rating}_bp"]
        for rating in ["A", "BBB", "BB"]
    )
    if not (table[weights].sum(axis=1) - 1.0).abs().lt(1e-9).all():
        raise ValueError("candidate rating weights must sum to one")
    return table


def main() -> None:
    """Print a compact chapter result summary."""
    print(exposure_summary().round(2))
    print(stress_losses().round(2))
    print(allocation_table().round(6))


if __name__ == "__main__":
    main()
