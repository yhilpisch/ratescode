"""
Python & AI for Rates, Bonds, and Credit
Chapter 22 · VaR, Expected Shortfall, and Stress Testing

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

VaR, expected shortfall, and stress figure for Chapter 22.
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
LIGHT_BLUE = "#8CB6E8"
ORANGE = "#D9822B"


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FACTOR_COLS = ["rate_2y_bp", "rate_5y_bp", "rate_10y_bp", "spread_bp"]


def exposures() -> pd.Series:
    """Load portfolio exposures by factor."""
    table = pd.read_csv(DATA / "ch22_portfolio_exposures.csv")
    return table.set_index("factor")["exposure"]


def losses() -> pd.Series:
    """Return positive historical loss numbers."""
    changes = pd.read_csv(DATA / "ch22_factor_changes.csv")
    values = (changes[FACTOR_COLS] * exposures()).sum(axis=1)
    return values.rename("loss")


def stress_losses() -> pd.Series:
    """Return positive stress loss numbers."""
    stress = pd.read_csv(DATA / "ch22_stress_scenarios.csv")
    values = (stress[FACTOR_COLS] * exposures()).sum(axis=1)
    return pd.Series(values.to_numpy(float), index=stress["scenario"])


def main() -> None:
    """Create the VaR, ES, and stress-test figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    loss = losses()
    var = float(np.quantile(loss, 0.95, method="lower"))
    es = float(loss[loss >= var].mean())
    stress = stress_losses()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    axes[0].hist(loss, bins=12, color=LIGHT_BLUE, alpha=0.75)
    axes[0].axvline(var, color=ORANGE, lw=2, label="VaR 95")
    axes[0].axvline(es, color=NAVY, lw=2, ls="--", label="ES 95")
    axes[0].set_title("Historical losses", color=NAVY)
    axes[0].set_xlabel("Loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(stress.index, stress.values, color=ORANGE)
    axes[1].set_title("Named stress losses", color=NAVY)
    axes[1].set_xlabel("Loss")
    axes[1].invert_yaxis()

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch22_var_es_loss_distribution.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
