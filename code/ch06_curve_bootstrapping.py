"""
Python & AI for Rates, Bonds, and Credit
Chapter 6 · Curve Bootstrapping and Multi-Curve Discounting

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


def bootstrap_discounts() -> pd.DataFrame:
    """Bootstrap annual par-swap discount factors from frozen quotes."""
    quotes = pd.read_csv(DATA / "ch06_swap_rates.csv")
    discounts: list[float] = []  # bootstrapped discount factors
    for _, row in quotes.iterrows():
        rate = float(row["swap_rate"])
        prior_annuity = sum(discounts)  # value of earlier fixed coupons
        discount = (1.0 - rate * prior_annuity) / (1.0 + rate)
        discounts.append(discount)
    quotes["discount"] = discounts
    quotes["zero_rate"] = -pd.Series(discounts).map(np.log) / quotes[
        "maturity"
    ]
    return quotes


def reprice_par_swaps() -> pd.DataFrame:
    """Check that bootstrapped discounts reprice the input swaps."""
    curve = bootstrap_discounts()
    rows = []  # repricing diagnostics
    for idx, row in curve.iterrows():
        rate = float(row["swap_rate"])
        fixed_leg = rate * curve.loc[:idx, "discount"].sum()
        float_leg = 1.0 - float(row["discount"])
        rows.append({
            "maturity": row["maturity"],
            "residual": fixed_leg - float_leg,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    curve = bootstrap_discounts()  # bootstrapped curve
    residuals = reprice_par_swaps()  # pricing check
    print(curve.round(6))
    print(residuals.round(10))


if __name__ == "__main__":
    main()
