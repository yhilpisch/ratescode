"""
Python & AI for Rates, Bonds, and Credit
Chapter 11 · Yield-Curve Dynamics and Factor Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Curve factor diagnostics figure for Chapter 11.
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
YIELD_COLS = ["y1", "y2", "y5", "y10"]
MATURITIES = np.array([1.0, 2.0, 5.0, 10.0])


def pca_loadings() -> np.ndarray:
    """Return the first three principal component loadings."""
    history = pd.read_csv(DATA / "ch11_yield_curve_history.csv")
    changes = history[YIELD_COLS].diff().dropna() * 10000.0
    centered = changes - changes.mean()  # demean changes
    cov = np.cov(centered.to_numpy(float), rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return vectors[:, order[:3]]


def ns_curve() -> np.ndarray:
    """Return a simple Nelson-Siegel fitted curve."""
    beta0, beta1, beta2, tau = 0.045, -0.015, -0.010, 2.5
    x = MATURITIES / tau
    slope = (1.0 - np.exp(-x)) / x
    curvature = slope - np.exp(-x)
    return beta0 + beta1 * slope + beta2 * curvature


def main() -> None:
    """Create the curve factor diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    loadings = pca_loadings()
    history = pd.read_csv(DATA / "ch11_yield_curve_history.csv")
    observed = history[YIELD_COLS].iloc[-1].to_numpy(float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    for idx, color in enumerate([BLUE, ORANGE, LIGHT_BLUE]):
        axes[0].plot(MATURITIES, loadings[:, idx], marker="o",
                     color=color, label=f"PC{idx + 1}")
    axes[0].set_title("PCA loadings", color=NAVY)
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Loading")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(MATURITIES, observed * 100, "o", color=BLUE,
                 label="observed")
    axes[1].plot(MATURITIES, ns_curve() * 100, color=ORANGE,
                 label="Nelson-Siegel")
    axes[1].set_title("Parametric curve fit", color=NAVY)
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("Yield percent")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch11_curve_factor_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
