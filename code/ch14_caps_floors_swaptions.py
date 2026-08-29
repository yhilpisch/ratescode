"""
Python & AI for Rates, Bonds, and Credit
Chapter 14 · Caps, Floors, and Swaptions

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

import importlib.util
from math import exp, log, sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_normal_cdf():
    """Load the shared normal CDF helper from the repository root."""
    spec = importlib.util.spec_from_file_location(
        "_common_math", ROOT / "common_math.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.normal_cdf


normal_cdf = load_normal_cdf()


def black_call(forward: float, strike: float, vol: float,
               expiry: float) -> float:
    """Return Black call value per unit annuity or accrual."""
    sigma_root = vol * sqrt(expiry)
    d1 = (log(forward / strike) + 0.5 * sigma_root ** 2) / sigma_root
    d2 = d1 - sigma_root
    return forward * normal_cdf(d1) - strike * normal_cdf(d2)


def caplet_values(notional: float=10_000_000.0) -> pd.DataFrame:
    """Compute Black caplet values from frozen inputs."""
    table = pd.read_csv(DATA / "ch14_caplet_inputs.csv")
    values = []  # caplet present values
    for _, row in table.iterrows():
        option = black_call(row["forward"], row["strike"],
                            row["vol"], row["expiry"])
        values.append(notional * row["discount"] * row["alpha"] * option)
    table["caplet_value"] = values
    return table


def cap_value() -> float:
    """Return the sum of caplet values."""
    return float(caplet_values()["caplet_value"].sum())


def payer_swaption_value() -> float:
    """Compute a payer swaption value from frozen Black inputs."""
    row = pd.read_csv(DATA / "ch14_swaption_inputs.csv").iloc[0]
    option = black_call(row["forward_swap"], row["strike"],
                        row["vol"], row["expiry"])
    return float(row["notional"] * row["annuity"] * option)


def main() -> None:
    """Print a compact chapter result summary."""
    caplets = caplet_values()
    print(caplets[["period", "caplet_value"]].round(2))
    print(f"cap_value: {cap_value():.2f}")
    print(f"payer_swaption: {payer_swaption_value():.2f}")


if __name__ == "__main__":
    main()
