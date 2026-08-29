"""
Python & AI for Rates, Bonds, and Credit
Chapter 23 · Model Risk, Calibration Risk, and Verification

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


def price_from_yield(yield_pa: float, freq: int=2) -> float:
    """Price the frozen bond cash flows from a yield."""
    flows = pd.read_csv(DATA / "ch23_bond_roundtrip.csv")
    times = flows["time"].to_numpy(float)  # payment times
    cashflows = flows["cashflow"].to_numpy(float)  # cash payments
    discount = (1.0 + yield_pa / freq) ** (-freq * times)
    return float(np.dot(cashflows, discount))


def yield_from_price(price: float, freq: int=2) -> float:
    """Recover yield by bisection from a target price."""
    low, high = 0.0, 0.20  # search bracket
    for _ in range(80):
        mid = 0.5 * (low + high)  # candidate yield
        if price_from_yield(mid, freq) > price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def roundtrip_check() -> dict[str, float]:
    """Check price/yield consistency for the frozen bond."""
    target_yield = 0.045  # test input yield
    price = price_from_yield(target_yield)
    recovered = yield_from_price(price)
    return {
        "price": price,
        "target_yield": target_yield,
        "recovered_yield": recovered,
        "yield_error": recovered - target_yield,
    }


def swap_repricing_residual() -> float:
    """Check that a par swap has near-zero value on its curve."""
    table = pd.read_csv(DATA / "ch23_swap_repricing.csv")
    annuity = float((table["alpha"] * table["discount"]).sum())  # PVBP
    par_rate = (1.0 - float(table["discount"].iloc[-1])) / annuity
    fixed_leg = par_rate * annuity  # fixed-leg value
    float_leg = 1.0 - float(table["discount"].iloc[-1])  # float value
    return fixed_leg - float_leg


def finite_difference_dv01() -> dict[str, float]:
    """Compare finite-difference and duration-style DV01 estimates."""
    ytm = 0.045  # base yield
    price = price_from_yield(ytm)
    bumped = price_from_yield(ytm + 0.0001)
    fd_dv01 = price - bumped  # finite-difference DV01
    flows = pd.read_csv(DATA / "ch23_bond_roundtrip.csv")
    times = flows["time"].to_numpy(float)
    cashflows = flows["cashflow"].to_numpy(float)
    discount = (1.0 + ytm / 2) ** (-2 * times)
    weights = cashflows * discount / price  # present-value weights
    mac_duration = float(np.dot(times, weights))
    mod_duration = mac_duration / (1.0 + ytm / 2)
    approx_dv01 = price * mod_duration / 10000.0
    return {"fd_dv01": fd_dv01, "approx_dv01": approx_dv01}


def calibration_residuals() -> pd.DataFrame:
    """Return residuals for two simple curve-fitting choices."""
    quotes = pd.read_csv(DATA / "ch23_curve_quotes.csv")
    quotes["linear_residual_bp"] = (
        quotes["model_linear"] - quotes["market_rate"]
    ) * 10000.0
    quotes["smooth_residual_bp"] = (
        quotes["model_smooth"] - quotes["market_rate"]
    ) * 10000.0
    return quotes[["maturity", "linear_residual_bp", "smooth_residual_bp"]]


def main() -> None:
    """Print a compact chapter result summary."""
    print({key: round(value, 8) for key, value in roundtrip_check().items()})
    print(f"swap_residual: {swap_repricing_residual():.10f}")
    print({key: round(value, 6) for key, value in
           finite_difference_dv01().items()})
    print(calibration_residuals().round(4))


if __name__ == "__main__":
    main()
