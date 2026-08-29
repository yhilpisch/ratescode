"""
Python & AI for Rates, Bonds, and Credit
Chapter 14 · Caps, Floors, and Swaptions

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Rates option surface figure for Chapter 14.
"""
from __future__ import annotations

import importlib.util
import os
from math import log, sqrt
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
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


def black_call(forward: float, strike: float, vol: float,
               expiry: float) -> float:
    """Return Black call value per unit notional."""
    root = vol * sqrt(expiry)
    d1 = (log(forward / strike) + 0.5 * root ** 2) / root
    d2 = d1 - root
    return forward * normal_cdf(d1) - strike * normal_cdf(d2)


def main() -> None:
    """Create the rates option surface figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    surface = pd.read_csv(DATA / "ch14_synthetic_vol_surface.csv")
    pivot = surface.pivot(index="expiry", columns="tenor", values="vol")
    strikes = np.linspace(0.025, 0.055, 15)
    values = [black_call(0.041, strike, 0.24, 1.0) for strike in strikes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    image = axes[0].imshow(pivot.to_numpy() * 100, cmap="Blues",
                           aspect="auto")
    axes[0].set_title("Synthetic volatility surface", color=NAVY)
    axes[0].set_xticks(np.arange(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(np.arange(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel("Tenor")
    axes[0].set_ylabel("Expiry")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(strikes * 100, np.array(values) * 10000, color=ORANGE)
    axes[1].set_title("Option value by strike", color=NAVY)
    axes[1].set_xlabel("Strike percent")
    axes[1].set_ylabel("Value bp")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch14_rates_option_surface.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
