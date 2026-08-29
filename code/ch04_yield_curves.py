"""
Python & AI for Rates, Bonds, and Credit
Chapter 4 · Yield Curves, Zero Rates, and Forward Rates

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


def load_curve() -> pd.DataFrame:
    """Load the frozen zero-curve nodes."""
    return pd.read_csv(DATA / "ch04_zero_curve.csv")


def curve_table() -> pd.DataFrame:
    """Return discount factors and one-period forward rates."""
    curve = load_curve()  # zero-rate nodes
    times = curve["maturity"].to_numpy(float)  # maturities
    rates = curve["zero_rate"].to_numpy(float)  # zero rates
    curve["discount"] = np.exp(-rates * times)  # zero discounts
    forwards = [np.nan]  # first node has no prior interval
    for left, right, rate_l, rate_r in zip(times[:-1], times[1:],
                                           rates[:-1], rates[1:]):
        numerator = rate_r * right - rate_l * left
        forwards.append(numerator / (right - left))  # interval forward
    curve["forward_rate"] = forwards
    return curve


def interpolate_discount(maturity: float=3.0) -> float:
    """Interpolate a continuously compounded zero discount factor."""
    curve = load_curve()
    rate = float(np.interp(maturity, curve["maturity"], curve["zero_rate"]))
    return float(np.exp(-rate * maturity))


def main() -> None:
    """Print a compact chapter result summary."""
    frame = curve_table()  # curve diagnostics
    discount = interpolate_discount()  # three-year discount factor
    print(frame.round(6))
    print(f"discount_3y: {discount:.6f}")


if __name__ == "__main__":
    main()
