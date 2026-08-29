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
    """Load the initial curve used for Hull-White-style fitting."""
    return pd.read_csv(DATA / "ch12_initial_zero_curve.csv")


def hull_white_target(step: int, dt: float) -> float:
    """Interpolate a deterministic target rate from the initial curve."""
    curve = initial_zero_curve()
    time = min((step + 1) * dt, float(curve["maturity"].max()))
    return float(np.interp(time, curve["maturity"], curve["zero_rate"]))


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
        rates[step + 1] = np.maximum(previous + drift + diffusion, -0.02)
    return rates


def zero_price_from_mean_path(model: str, maturity: float=1.0) -> float:
    """Approximate a zero-coupon price from the average short-rate path.

    This uses the path-averaged short rate rather than integrating along
    each path; it is a pedagogical simplification of the exact
    discount-factor average exp(-cumsum(r*dt)).
    """
    paths = simulate_paths(model)
    mean_rate = float(paths.mean(axis=1).mean())  # average short rate
    return float(np.exp(-mean_rate * maturity))


def model_summary() -> pd.DataFrame:
    """Summarise simulated short-rate path statistics."""
    rows = []  # model diagnostics
    for model in load_params()["model"]:
        paths = simulate_paths(model)
        rows.append({
            "model": model,
            "mean_final": float(paths[-1].mean()),
            "min_rate": float(paths.min()),
            "zero_1y": zero_price_from_mean_path(model),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(model_summary().round(5))


if __name__ == "__main__":
    main()
