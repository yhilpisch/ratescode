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
    """Simulate terminal curves with a shared one-factor Gaussian shock."""
    inputs = load_inputs()
    base = inputs["forward_rate"].to_numpy(float)
    vol = inputs["volatility"].to_numpy(float)
    rng = np.random.default_rng(1300)
    shocks = rng.standard_normal((paths, 1))
    return base + vol * shocks * np.sqrt(0.5)


def draw_scenarios(ax) -> None:
    """Draw unpriced six-month terminal scenarios with common shocks."""
    inputs = load_inputs()
    maturities = inputs["maturity"].to_numpy(float)
    for idx, path in enumerate(simulate_paths()[:8]):
        ax.plot(maturities, path * 100, color=LIGHT_BLUE, alpha=0.8,
                label="Six-month scenarios" if idx == 0 else None)
    ax.plot(maturities, inputs["forward_rate"] * 100, color=NAVY,
            lw=2, label="Initial snapshot")
    ax.set(xlabel="Fixed maturity from today (years)",
           ylabel="Forward rate (%)")
    ax.legend(frameon=False, fontsize=10)


def draw_volatility(ax) -> None:
    """Draw absolute Gaussian volatilities, not Black volatilities."""
    inputs = load_inputs()
    ax.bar(inputs["maturity"], inputs["volatility"] * 10000,
           width=0.3, color=ORANGE)
    ax.set(xlabel="Fixed maturity from today (years)",
           ylabel="Absolute volatility (bp/sqrt(year))")


def main() -> None:
    """Create a book composite and separate slide diagnostics."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    draws = [draw_scenarios, draw_volatility]
    for ax, draw in zip(axes, draws):
        draw(ax)
        ax.tick_params(colors=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ch13_forward_curve_dynamics.pdf")
    plt.close(fig)
    for suffix, draw in zip(["scenarios", "volatility"], draws):
        fig, ax = plt.subplots(figsize=(7.8, 2.7))
        draw(ax)
        ax.tick_params(colors=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"ch13_forward_{suffix}_slides.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
