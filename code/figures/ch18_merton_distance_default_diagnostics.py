"""
Python & AI for Rates, Bonds, and Credit
Chapter 18 · Structural Credit Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Merton distance-to-default diagnostics figure for Chapter 18.
"""
from __future__ import annotations

import importlib.util
import os
from math import log, sqrt
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
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


def metrics(asset_value: float, vol: float) -> tuple[float, float]:
    """Return distance to default and default probability."""
    debt, rate, maturity = 100.0, 0.04, 1.0
    numerator = log(asset_value / debt) + (rate - 0.5 * vol ** 2) * maturity
    distance = numerator / (vol * sqrt(maturity))
    return distance, normal_cdf(-distance)


def main() -> None:
    """Create the Merton distance-to-default diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(DATA / "ch18_merton_scenarios.csv")
    values = [metrics(row.asset_value, row.asset_volatility)
              for row in scenarios.itertuples()]
    scenarios["distance"] = [value[0] for value in values]
    scenarios["default_probability"] = [value[1] for value in values]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(scenarios["scenario"], scenarios["distance"], color=BLUE)
    axes[0].set_title("Distance to default", color=NAVY)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(scenarios["scenario"],
                scenarios["default_probability"] * 100, color=ORANGE)
    axes[1].set_title("Default probability", color=NAVY)
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch18_merton_distance_default_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
