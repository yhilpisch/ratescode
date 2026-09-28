"""
Python & AI for Rates, Bonds, and Credit
Chapter 13 · HJM and LMM Perspectives

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


def load_forward_inputs() -> pd.DataFrame:
    """Load frozen forward rates and volatility assumptions."""
    curve = pd.read_csv(DATA / "ch13_forward_curve_snapshot.csv")
    vol = pd.read_csv(DATA / "ch13_forward_vol_assumptions.csv")
    return curve.merge(vol, on="maturity")


def hjm_drift() -> pd.DataFrame:
    """Compute a simple one-factor HJM drift proxy."""
    inputs = load_forward_inputs()
    dtau = inputs["maturity"].diff().fillna(inputs["maturity"].iloc[0])
    cumulative = (inputs["volatility"] * dtau).cumsum()  # tenor-scaled vol
    inputs["drift_proxy"] = inputs["volatility"] * cumulative
    return inputs[["maturity", "forward_rate", "drift_proxy"]]


def simulate_forward_paths(steps: int=6,
                           paths: int=3) -> pd.DataFrame:
    """Simulate a one-factor HJM-style sketch with common tenor shocks.

    The drift proxy is not a calibrated, measure-validated HJM specification
    and must not be used for derivative valuation.
    """
    inputs = load_forward_inputs()
    drift = hjm_drift()["drift_proxy"].to_numpy(float)
    volatility = inputs["volatility"].to_numpy(float)
    rng = np.random.default_rng(1300)  # deterministic seed
    dt = 0.25
    rows = []  # simulated terminal forwards
    for path in range(paths):
        rates = inputs["forward_rate"].to_numpy(float).copy()
        for _ in range(steps):
            shock = rng.standard_normal()  # common one-factor shock by tenor
            # HJM: additive evolution in forward rate
            rates += volatility * np.sqrt(dt) * shock
            rates += drift * dt
        for maturity, rate in zip(inputs["maturity"], rates, strict=True):
            rows.append({"path": path, "maturity": maturity, "rate": rate})
    return pd.DataFrame(rows)


def lmm_style_step() -> pd.DataFrame:
    """Apply one lognormal market-forward-rate step (LMM style)."""
    inputs = load_forward_inputs()
    dt = 0.25
    shock = np.array([-0.5, -0.2, 0.1, 0.3, 0.2, 0.0])
    vol = inputs["volatility"].to_numpy(float)
    rates = inputs["forward_rate"].to_numpy(float)
    evolved = rates * np.exp(-0.5 * vol ** 2 * dt + vol * np.sqrt(dt) * shock)
    return pd.DataFrame({"maturity": inputs["maturity"], "evolved": evolved})


def main() -> None:
    """Print a compact chapter result summary."""
    print(hjm_drift().round(6))
    print(simulate_forward_paths().head(9).round(5))
    print(lmm_style_step().round(6))


if __name__ == "__main__":
    main()
