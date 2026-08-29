"""
Python & AI for Rates, Bonds, and Credit
Chapter 3 · Bond Mathematics and Market Conventions

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def actual_360(start: date, end: date) -> float:
    """Return an ACT/360 accrual fraction."""
    days = (end - start).days  # actual calendar days
    return days / 360.0


def thirty_360(start: date, end: date) -> float:
    """Return a simple 30/360 accrual fraction."""
    days = 360 * (end.year - start.year)  # year component
    days += 30 * (end.month - start.month)  # month component
    days += min(end.day, 30) - min(start.day, 30)  # day component
    return days / 360.0


def accrued_interest() -> dict[str, float]:
    """Compute accrued interest under two day-count conventions."""
    last_coupon = date(2026, 1, 15)  # previous coupon date
    settlement = date(2026, 3, 10)  # settlement date
    annual_coupon_cash = 5.00  # annual coupon cash
    return {
        "act_360": annual_coupon_cash * actual_360(last_coupon, settlement),
        "30_360": annual_coupon_cash * thirty_360(last_coupon, settlement),
    }


def bond_risk(yield_pa: float=0.04, freq: int=2) -> dict[str, float]:
    """Compute price and duration for a stylised coupon bond."""
    frame = pd.read_csv(DATA / "ch03_bond_cashflows.csv")
    times = frame["time"].to_numpy(float)  # coupon times
    cashflows = frame["cashflow"].to_numpy(float)  # payments
    discount = (1.0 + yield_pa / freq) ** (-freq * times)
    pv_parts = cashflows * discount  # value contributions
    price = float(pv_parts.sum())  # dirty value
    weights = pv_parts / price  # present-value weights
    mac_duration = float(np.dot(times, weights))  # time-weighted value
    mod_duration = mac_duration / (1.0 + yield_pa / freq)  # yield sensitivity
    return {
        "price": price,
        "mac_duration": mac_duration,
        "mod_duration": mod_duration,
    }


def main() -> None:
    """Print a compact chapter result summary."""
    accrued = accrued_interest()  # accrued-interest comparison
    risk = bond_risk()  # price and duration output
    print({key: round(value, 4) for key, value in accrued.items()})
    print({key: round(value, 4) for key, value in risk.items()})


if __name__ == "__main__":
    main()
