"""
Python & AI for Rates, Bonds, and Credit
Chapter 9 · Interest-Rate Swaps and Swap Risk

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


def load_inputs() -> pd.DataFrame:
    """Load frozen swap discount and forward inputs."""
    return pd.read_csv(DATA / "ch09_swap_inputs.csv")


def par_swap_rate() -> float:
    """Compute the fixed rate that makes the swap value zero."""
    table = load_inputs()
    float_leg = 1.0 - float(table["discount"].iloc[-1])  # float-leg value
    annuity = float((table["alpha"] * table["discount"]).sum())  # PVBP
    return float_leg / annuity  # par fixed rate


def swap_value(fixed_rate: float=0.04,
               notional: float=100_000_000.0) -> float:
    """Value a receive-fixed swap against the frozen curve."""
    table = load_inputs()
    annuity = float((table["alpha"] * table["discount"]).sum())  # PVBP
    float_leg = 1.0 - float(table["discount"].iloc[-1])  # float value
    fixed_leg = fixed_rate * annuity  # fixed value
    return notional * (fixed_leg - float_leg)  # receive-fixed value


def swap_dv01(fixed_rate: float=0.04) -> float:
    """Approximate signed DV01 from a rough one-bp discount bump."""
    base = load_inputs()
    bumped = base.copy()  # simple parallel discount shock
    bumped["discount"] *= 1.0 - 0.0001 * bumped["period"]
    annuity = float((bumped["alpha"] * bumped["discount"]).sum())  # PVBP
    float_leg = 1.0 - float(bumped["discount"].iloc[-1])  # shocked float
    fixed_leg = fixed_rate * annuity  # shocked fixed
    bumped_value = 100_000_000.0 * (fixed_leg - float_leg)  # shocked value
    return swap_value(fixed_rate) - bumped_value  # signed DV01


def summary_table() -> pd.DataFrame:
    """Return a compact swap valuation summary."""
    return pd.DataFrame([{
        "par_rate": par_swap_rate(),
        "value_4pct": swap_value(),
        "dv01": swap_dv01(),
    }])


def main() -> None:
    """Print a compact chapter result summary."""
    print(summary_table().round(4))


if __name__ == "__main__":
    main()
