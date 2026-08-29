"""
Python & AI for Rates, Bonds, and Credit
Chapter 18 · Structural Credit Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

import importlib.util
from math import log, sqrt
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


def inputs() -> dict[str, float]:
    """Load frozen Merton model inputs."""
    table = pd.read_csv(DATA / "ch18_merton_inputs.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def merton_metrics(asset_value: float,
                   asset_volatility: float) -> dict[str, float]:
    """Compute distance to default and default probability."""
    params = inputs()
    debt = params["debt_face"]
    rate = params["risk_free_rate"]
    maturity = params["maturity"]
    numerator = log(asset_value / debt)
    numerator += (rate - 0.5 * asset_volatility ** 2) * maturity
    denominator = asset_volatility * sqrt(maturity)
    distance = numerator / denominator
    default_probability = normal_cdf(-distance)
    return {
        "distance_to_default": distance,
        "default_probability": default_probability,
    }


def scenario_table() -> pd.DataFrame:
    """Evaluate Merton metrics across frozen scenarios."""
    scenarios = pd.read_csv(DATA / "ch18_merton_scenarios.csv")
    rows = []  # scenario diagnostics
    for _, row in scenarios.iterrows():
        metrics = merton_metrics(row["asset_value"], row["asset_volatility"])
        rows.append({"scenario": row["scenario"], **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(scenario_table().round(6))


if __name__ == "__main__":
    main()
