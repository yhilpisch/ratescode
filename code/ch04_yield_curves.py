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
    """Return node discounts and continuously compounded interval forwards."""
    curve = load_curve()  # zero-rate nodes
    times = curve["maturity"].to_numpy(float)  # maturities
    rates = curve["zero_rate"].to_numpy(float)  # zero rates
    curve["discount"] = np.exp(-rates * times)  # zero discounts
    forwards = [np.nan]  # first node has no prior interval
    for left, right, rate_l, rate_r in zip(times[:-1], times[1:],
                                           rates[:-1], rates[1:]):
        numerator = rate_r * right - rate_l * left
        forwards.append(numerator / (right - left))  # interval forward
    curve["forward_cc"] = forwards
    return curve


def interpolate_discount(maturity: float=3.0) -> float:
    """Interpolate log discounts inside the supplied maturity range."""
    curve = load_curve()
    times = curve["maturity"].to_numpy(float)
    if not np.isfinite(maturity) or not times[0] <= maturity <= times[-1]:
        raise ValueError("maturity must be finite and within the curve nodes")
    log_discounts = -curve["zero_rate"].to_numpy(float) * times
    log_discount = np.interp(maturity, times, log_discounts)
    return float(np.exp(log_discount))


def main() -> None:
    """Print a compact chapter result summary."""
    frame = curve_table()  # curve diagnostics
    discount = interpolate_discount()  # log-discount interpolated 3Y factor
    print(frame.round(6))
    print(f"discount_3y: {discount:.6f}")


if __name__ == "__main__":
    main()
