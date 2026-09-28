"""
Python & AI for Rates, Bonds, and Credit
Chapter 12 · Short-Rate Models: Vasicek, CIR, and Hull-White

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


def load_params() -> pd.DataFrame:
    """Load frozen short-rate model parameters."""
    return pd.read_csv(DATA / "ch12_short_rate_params.csv")


def initial_zero_curve() -> pd.DataFrame:
    """Load the curve used by the Hull-White-style target proxy."""
    return pd.read_csv(DATA / "ch12_initial_zero_curve.csv")


def hull_white_target(step: int, dt: float) -> float:
    """Return the zero-rate proxy used by the Hull-White-style toy process.

    This is not the time-dependent Hull-White drift parameter and does not
    make the simulated process fit the initial discount curve. Flat endpoint
    extension is explicit because early monthly steps precede the first node.
    """
    curve = initial_zero_curve()
    time = (step + 1) * dt
    maturities = curve["maturity"].to_numpy(float)
    rates = curve["zero_rate"].to_numpy(float)
    return float(np.interp(time, maturities, rates,
                           left=rates[0], right=rates[-1]))


def simulate_paths(model: str, steps: int=12, paths: int=4) -> np.ndarray:
    """Simulate short-rate paths for one model."""
    params = load_params().set_index("model").loc[model]
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(params["sigma"])
    r0 = float(params["r0"])
    dt = 1.0 / 12.0
    rng = np.random.default_rng(100 + len(model))  # deterministic seed
    rates = np.full((steps + 1, paths), r0)  # path matrix
    for step in range(steps):
        shock = rng.standard_normal(paths)
        previous = rates[step]
        if model == "cir":
            root = np.sqrt(np.maximum(previous, 0.0))
            diffusion = sigma * root * np.sqrt(dt) * shock
            target = theta
        else:
            diffusion = sigma * np.sqrt(dt) * shock
            if model == "hull-white":
                target = hull_white_target(step, dt)
            else:
                target = theta
        drift = kappa * (target - previous) * dt
        candidate = previous + drift + diffusion
        if model == "cir":
            rates[step + 1] = np.maximum(candidate, 0.0)  # projected Euler step
        else:
            rates[step + 1] = candidate  # normal models may be negative
    return rates


def zero_coupon_price_mc(
    model: str,
    maturity: float=1.0,
    paths: int=10_000,
) -> tuple[float, float]:
    """Estimate a zero price and sampling error by discounting each path.

    The standard error does not include monthly time-discretization or model
    error.
    """
    dt = 1.0 / 12.0
    steps = int(round(maturity / dt))
    if maturity <= 0.0 or not np.isclose(steps * dt, maturity):
        raise ValueError("maturity must be a positive whole number of months")
    if paths < 2:
        raise ValueError("at least two paths are required")
    rate_paths = simulate_paths(model, steps=steps, paths=paths)
    integrals = rate_paths[:-1].sum(axis=0) * dt  # left-endpoint path integral
    discounts = np.exp(-integrals)  # average pathwise discount factors
    price = float(discounts.mean())
    standard_error = float(discounts.std(ddof=1) / np.sqrt(paths))
    return price, standard_error


def model_summary() -> pd.DataFrame:
    """Summarise simulated short-rate path statistics."""
    rows = []  # model diagnostics
    for model in load_params()["model"]:
        paths = simulate_paths(model)
        zero_price, zero_se = zero_coupon_price_mc(model)
        rows.append({
            "model": model,
            "mean_final": float(paths[-1].mean()),
            "min_rate": float(paths.min()),
            "zero_1y_mc": zero_price,
            "zero_1y_mc_se": zero_se,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(model_summary().round(5))


if __name__ == "__main__":
    main()
