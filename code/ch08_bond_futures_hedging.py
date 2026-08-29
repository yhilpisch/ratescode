"""
Python & AI for Rates, Bonds, and Credit
Chapter 8 · Bond Futures, CTD Bonds, and Duration Hedging

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


def contract_parameters() -> dict[str, float]:
    """Load the frozen futures-contract parameters."""
    table = pd.read_csv(DATA / "ch08_futures_contract.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def portfolio_dv01() -> float:
    """Compute the portfolio DV01 from value and duration inputs."""
    table = pd.read_csv(DATA / "ch08_futures_hedge_inputs.csv")
    value = table["value"]  # market values
    duration = table["modified_duration"]  # modified durations
    table["dv01"] = value * duration / 10000.0  # value per bp
    return float(table["dv01"].sum())


def hedge_ratio() -> float:
    """Return the number of futures contracts for a DV01 hedge."""
    params = contract_parameters()
    ctd_dv01 = params["ctd_dv01"]  # CTD bond DV01
    futures_dv01 = ctd_dv01 / params["conversion_factor"]  # futures DV01
    return portfolio_dv01() / futures_dv01


def hedge_pnl() -> pd.DataFrame:
    """Compare unhedged and hedged price moves for one rate shock."""
    params = contract_parameters()
    shock_bp = params["shock_bp"]  # parallel rate shock
    ctd_dv01 = params["ctd_dv01"]  # CTD bond DV01
    futures_dv01 = ctd_dv01 / params["conversion_factor"]  # futures DV01
    contracts = hedge_ratio()  # DV01-neutral contract count
    portfolio_loss = -portfolio_dv01() * shock_bp  # bond loss
    futures_gain = contracts * futures_dv01 * shock_bp  # hedge gain
    return pd.DataFrame([{
        "shock_bp": shock_bp,
        "portfolio_pnl": portfolio_loss,
        "futures_pnl": futures_gain,
        "net_pnl": portfolio_loss + futures_gain,
    }])


def main() -> None:
    """Print a compact chapter result summary."""
    print(f"portfolio_dv01: {portfolio_dv01():.2f}")
    print(f"hedge_contracts: {hedge_ratio():.2f}")
    print(hedge_pnl().round(2))


if __name__ == "__main__":
    main()
