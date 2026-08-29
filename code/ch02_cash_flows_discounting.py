"""
Python & AI for Rates, Bonds, and Credit
Chapter 2 · Cash Flows, Discounting, and Present Value

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


def load_cashflows() -> pd.DataFrame:
    """Load the frozen toy cash-flow snapshot."""
    return pd.read_csv(DATA / "ch02_cashflows.csv")


def present_value(rate: float=0.04) -> pd.DataFrame:
    """Return discount factors and present-value contributions."""
    frame = load_cashflows()  # stored cash-flow table
    times = frame["time"].to_numpy(float)  # payment times
    cashflows = frame["cashflow"].to_numpy(float)  # payments
    frame["discount"] = np.exp(-rate * times)  # discount factors
    frame["pv_part"] = cashflows * frame["discount"]  # contributions
    return frame


def compounding_comparison(
    rate: float=0.04,
    time: float=2.0,
) -> dict[str, float]:
    """Compare simple, annual, and continuous discount factors."""
    simple = 1.0 / (1.0 + rate * time)  # simple discount
    annual = 1.0 / (1.0 + rate) ** int(time)  # annual discount
    continuous = float(np.exp(-rate * time))  # continuous discount
    return {
        "simple": simple,
        "annual": annual,
        "continuous": continuous,
    }


def main() -> None:
    """Print a compact chapter result summary."""
    frame = present_value()  # valuation table
    price = float(frame["pv_part"].sum())  # total present value
    factors = compounding_comparison()  # convention comparison
    print(frame.round({"discount": 6, "pv_part": 4}))
    print(f"price: {price:.4f}")
    print({key: round(value, 6) for key, value in factors.items()})


if __name__ == "__main__":
    main()
