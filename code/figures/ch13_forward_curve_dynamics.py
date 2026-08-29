"""
Python & AI for Rates, Bonds, and Credit
Chapter 13 · HJM and LMM Perspectives

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Forward curve dynamics figure for Chapter 13.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_inputs() -> pd.DataFrame:
    """Load forward rates and volatilities."""
    curve = pd.read_csv(DATA / "ch13_forward_curve_snapshot.csv")
    vol = pd.read_csv(DATA / "ch13_forward_vol_assumptions.csv")
    return curve.merge(vol, on="maturity")


def simulate_paths(paths: int=20) -> np.ndarray:
    """Simulate simplified terminal forward curves."""
    inputs = load_inputs()
    base = inputs["forward_rate"].to_numpy(float)
    vol = inputs["volatility"].to_numpy(float)
    rng = np.random.default_rng(1300)
    shocks = rng.standard_normal((paths, len(base)))
    return base + vol * shocks * np.sqrt(0.5)


def main() -> None:
    """Create the forward curve dynamics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    maturities = inputs["maturity"].to_numpy(float)
    paths = simulate_paths()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(maturities, inputs["forward_rate"] * 100,
                 color=NAVY, lw=2, label="initial")
    for path in paths[:8]:
        axes[0].plot(maturities, path * 100, color=LIGHT_BLUE, alpha=0.55)
    axes[0].set_title("Forward-curve paths", color=NAVY)
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Forward rate percent")

    axes[1].bar(maturities, inputs["volatility"] * 10000, color=ORANGE)
    axes[1].set_title("Volatility assumptions", color=NAVY)
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("Volatility bp")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch13_forward_curve_dynamics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
