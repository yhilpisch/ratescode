"""
Python & AI for Rates, Bonds, and Credit
Chapter 5 · Money Markets, Short Rates, and Overnight Indexing

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


def compounded_overnight() -> float:
    """Compound frozen overnight fixings over ACT/360 days."""
    frame = pd.read_csv(DATA / "ch05_overnight_rates.csv")
    rates = frame["overnight_rate"].to_numpy(float)  # daily fixings
    factors = 1.0 + rates / 360.0  # one-day gross factors
    return float(np.prod(factors) - 1.0)


def money_market_discounts() -> pd.DataFrame:
    """Convert simple money-market quotes into discount factors."""
    quotes = pd.read_csv(DATA / "ch05_money_market_quotes.csv")
    rates = quotes["rate"].to_numpy(float)  # quoted simple rates
    tau = quotes["tau"].to_numpy(float)  # year fractions
    quotes["discount"] = 1.0 / (1.0 + rates * tau)  # simple discount
    return quotes


def implied_forward() -> float:
    """Return a forward rate from two money-market discounts."""
    table = money_market_discounts()
    d_short, d_long = table["discount"].to_numpy(float)
    tau_short, tau_long = table["tau"].to_numpy(float)
    return float((d_short / d_long - 1.0) / (tau_long - tau_short))


def main() -> None:
    """Print a compact chapter result summary."""
    overnight = compounded_overnight()  # overnight return
    table = money_market_discounts()  # discount conversion
    forward = implied_forward()  # 3M-to-6M forward
    print(table.round(6))
    print(f"overnight_return: {overnight:.8f}")
    print(f"forward_3m_6m: {forward:.6f}")


if __name__ == "__main__":
    main()
