"""
Python & AI for Rates, Bonds, and Credit
Chapter 7 · Government Bonds and Sovereign Curve Risk

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


def zero_rate(maturity: float, curve: pd.DataFrame) -> float:
    """Interpolate the zero rate for one maturity."""
    return float(np.interp(maturity, curve["maturity"], curve["zero_rate"]))


def bond_cashflows(coupon: float, maturity: int) -> pd.DataFrame:
    """Return annual coupon cash flows for a par-100 notional bond."""
    times = np.arange(1, maturity + 1, dtype=float)  # annual cash-flow grid
    cashflows = np.full_like(times, coupon * 100.0)  # coupon cash flows
    cashflows[-1] += 100.0  # principal repayment
    return pd.DataFrame({"time": times, "cashflow": cashflows})


def price_bond(coupon: float, maturity: int, curve: pd.DataFrame) -> float:
    """Price a sovereign bond from interpolated zero rates."""
    flows = bond_cashflows(coupon, maturity)
    rates = [zero_rate(time, curve) for time in flows["time"]]
    discounts = np.exp(-np.array(rates) * flows["time"].to_numpy(float))
    return float(np.dot(flows["cashflow"], discounts))


def price_table() -> pd.DataFrame:
    """Return prices for the frozen sovereign bond universe."""
    curve = pd.read_csv(DATA / "ch07_sovereign_curve.csv")
    bonds = pd.read_csv(DATA / "ch07_sovereign_bonds.csv")
    prices = []  # curve-implied clean prices
    for _, row in bonds.iterrows():
        prices.append(price_bond(row["coupon"], int(row["maturity"]), curve))
    bonds["price"] = prices
    return bonds


def key_rate_dv01(node: float=5.0, shift_bp: float=1.0) -> pd.DataFrame:
    """Compute a finite-difference DV01 for one sovereign curve node."""
    curve = pd.read_csv(DATA / "ch07_sovereign_curve.csv")
    bumped = curve.copy()  # local shocked curve
    mask = bumped["maturity"] == node
    bumped.loc[mask, "zero_rate"] += shift_bp / 10000.0
    bonds = pd.read_csv(DATA / "ch07_sovereign_bonds.csv")
    rows = []  # key-rate risk rows
    for _, row in bonds.iterrows():
        base = price_bond(row["coupon"], int(row["maturity"]), curve)
        shock = price_bond(row["coupon"], int(row["maturity"]), bumped)
        rows.append({"bond": row["bond"], "dv01": base - shock})
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(price_table().round({"price": 4}))
    print(key_rate_dv01().round({"dv01": 5}))


if __name__ == "__main__":
    main()
