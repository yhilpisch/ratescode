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
    """Return undiscounted Black call value in rate units."""
    root = vol * sqrt(expiry)
    d1 = (log(forward / strike) + 0.5 * root ** 2) / root
    d2 = d1 - root
    return forward * normal_cdf(d1) - strike * normal_cdf(d2)


def draw_surface(ax) -> None:
    """Draw the synthetic expiry/tenor slice of Black volatility."""
    surface = pd.read_csv(DATA / "ch14_synthetic_vol_surface.csv")
    pivot = surface.pivot(index="expiry", columns="tenor", values="vol")
    image = ax.imshow(pivot.to_numpy() * 100, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set(xlabel="Underlying swap tenor (years)",
           ylabel="Option expiry (years)")
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iloc[row, col] * 100
            color = "white" if value > 24 else NAVY
            ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                    color=color, fontsize=11)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Relative Black volatility (%)")


def draw_strikes(ax) -> None:
    """Draw an unscaled Black call, excluding notional and annuity."""
    strikes = np.linspace(0.025, 0.055, 150)
    values = [black_call(0.041, strike, 0.24, 1.0) for strike in strikes]
    ax.plot(strikes * 100, np.array(values) * 10000, color=ORANGE, lw=2)
    ax.axvline(4.1, color=NAVY, ls="--", lw=1, label="Forward = 4.10%")
    ax.set(xlabel="Strike (%)", ylabel="Black call (bp of rate)")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Create the composite and single-panel slide figures."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    draws = [draw_surface, draw_strikes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, draw in zip(axes, draws):
        draw(ax)
        ax.tick_params(colors=NAVY)
    fig.tight_layout()
    fig.savefig(out / "ch14_rates_option_surface.pdf")
    plt.close(fig)
    for suffix, draw in zip(["volatility", "strike"], draws):
        fig, ax = plt.subplots(figsize=(7.8, 2.7))
        draw(ax)
        ax.tick_params(colors=NAVY)
        fig.tight_layout()
        fig.savefig(out / f"ch14_option_{suffix}_slides.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
