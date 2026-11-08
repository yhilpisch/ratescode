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
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from ch11_yield_curve_dynamics_factor_models import pca_summary

NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
DATA = ROOT / "data"
YIELD_COLS = ["y1", "y2", "y5", "y10"]
MATURITIES = np.array([1.0, 2.0, 5.0, 10.0])
TAU = 2.5  # fixed decay scale in years


def ns_design(maturities: np.ndarray) -> np.ndarray:
    """Build the Nelson-Siegel design with a predeclared decay scale."""
    x = maturities / TAU
    slope = -np.expm1(-x) / x
    return np.column_stack((np.ones_like(x), slope, slope - np.exp(-x)))


def fit_coefficients() -> np.ndarray:
    """Fit three coefficients to four synthetic nodes at fixed tau."""
    history = pd.read_csv(DATA / "ch11_yield_curve_history.csv")
    observed = history[YIELD_COLS].iloc[-1].to_numpy(float)
    return np.linalg.lstsq(ns_design(MATURITIES), observed, rcond=None)[0]


def ns_curve(maturities: np.ndarray=MATURITIES) -> np.ndarray:
    """Return the least-squares fitted curve, with tau fixed at 2.5 years."""
    return ns_design(maturities) @ fit_coefficients()


def pca_loadings() -> np.ndarray:
    """Return the canonical first three principal component loadings."""
    return pca_summary()[YIELD_COLS].to_numpy(float).T


def draw_loadings(ax) -> None:
    """Draw loading shapes and their explained variance shares."""
    summary = pca_summary()
    for idx, color in enumerate([BLUE, ORANGE, LIGHT_BLUE]):
        row = summary.iloc[idx]
        label = f"PC{idx + 1}: {row.explained:.2%}"
        ax.plot(MATURITIES, row[YIELD_COLS].to_numpy(float), marker="o",
                color=color, label=label, lw=2)
    ax.axhline(0, color=NAVY, lw=0.7)
    ax.set(xlabel="Maturity (years)", ylabel="Unit-vector loading")
    ax.legend(frameon=False, fontsize=11, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3, columnspacing=1.0,
              handlelength=1.8)


def draw_fit(ax) -> None:
    """Draw fitted yields and signed residuals in basis points."""
    observed = pd.read_csv(DATA / "ch11_yield_curve_history.csv")
    yields = observed[YIELD_COLS].iloc[-1].to_numpy(float)
    grid = np.linspace(1, 10, 200)
    ax.plot(grid, ns_curve(grid) * 100, color=ORANGE, lw=2,
            label="Least squares; fixed decay = 2.5 years")
    ax.scatter(MATURITIES, yields * 100, color=BLUE, s=55, zorder=3,
               label="Synthetic zero-rate nodes")
    residuals = (ns_curve() - yields) * 10000
    for maturity, rate, residual in zip(MATURITIES, yields, residuals):
        ax.annotate(f"{residual:+.2f} bp", (maturity, rate * 100),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", color=NAVY, fontsize=11)
    ax.set(xlabel="Maturity (years)", ylabel="Zero rate (%)",
           xlim=(0.5, 10.7), ylim=(3.4, 4.65))
    ax.legend(frameon=False, fontsize=10, loc="lower right")


def style_axis(ax) -> None:
    """Apply the chapter palette and remove unused spines."""
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Create the book composite and readable single-panel slide assets."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, draw, title in zip(axes, [draw_loadings, draw_fit],
                               ["PCA loadings", "Nelson-Siegel fit"]):
        draw(ax)
        style_axis(ax)
        ax.set_title(title, color=NAVY, pad=34)
    fig.tight_layout()
    fig.savefig(out / "ch11_curve_factor_diagnostics.pdf")
    plt.close(fig)
    for suffix, draw in [("pca", draw_loadings), ("fit", draw_fit)]:
        fig, ax = plt.subplots(figsize=(7.8, 2.7))
        draw(ax)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(out / f"ch11_curve_factor_{suffix}_slides.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
