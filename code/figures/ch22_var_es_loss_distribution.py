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
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from credit_figure_layout import (
    NAVY, BLUE, ORANGE, export_panels, legend_above,
)

import matplotlib.dates as mdates
from statistics import NormalDist
from ch22_var_expected_shortfall_stress_testing import (
    historical_losses, var_es, stress_losses, var_backtest,
)


def draw_histogram(ax) -> None:
    """Keep the canonical 40-row loss panel and tail conventions."""
    losses = historical_losses() / 1000
    metrics = var_es()
    ax.hist(losses, bins=12, color="#8CB6E8", alpha=0.85)
    ax.axvline(metrics["var"] / 1000, color=ORANGE, lw=2,
               label="95% lower VaR")
    ax.axvline(metrics["es"] / 1000, color=NAVY, lw=2, ls="--",
               label="Mean of worst 2")
    ax.set(xlabel="One-day model loss ($ thousands)", ylabel="Observations")
    legend_above(ax)


def draw_stress(ax) -> None:
    """Show all four named hypothetical stress vectors."""
    table = stress_losses()
    bars = ax.barh(table.scenario, table.loss / 1000, color=ORANGE)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=NAVY)
    ax.invert_yaxis()
    ax.set(xlabel="Named stress loss ($ thousands)", xlim=(0, 335))


def draw_backtest(ax) -> None:
    """Use strictly prior expanding windows and mark replay exceptions."""
    table = var_backtest()
    ax.plot(table.date, table.loss / 1000, marker="o", ms=3,
            color=BLUE, label="Next-day replay loss")
    ax.step(table.date, table["var"] / 1000, where="mid", color=NAVY,
            lw=1.7, label="Prior-window 95% VaR")
    tail = table.loc[table.exception]
    ax.scatter(tail.date, tail.loss / 1000, color=ORANGE, marker="^",
               s=55, zorder=4, label="Exception")
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set(xlabel="Synthetic observation date (2025)",
           ylabel="Loss ($ thousands)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    legend_above(ax, columns=3)


def draw_tail_schematic(ax) -> None:
    """Illustrate a population quantile versus an upper-tail mean."""
    x = np.linspace(-3.5, 4, 500)
    density = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    z = NormalDist().inv_cdf(0.95)
    tail_mean = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi) / 0.05
    ax.plot(x, density, color=BLUE, lw=2)
    ax.fill_between(x, density, where=x >= z, color=ORANGE, alpha=0.45)
    ax.axvline(z, color=ORANGE, lw=2, label="95% quantile = 1.645")
    ax.axvline(tail_mean, color=NAVY, lw=2, ls="--",
                label="Upper-tail mean = 2.063")
    ax.text(2.5, 0.15, "Upper 5%", color=NAVY, ha="center")
    ax.set(xlabel="Standardised loss Z (illustrative standard normal)",
           ylabel="Probability density", xlim=(-3.5, 4), ylim=(0, 0.43))
    legend_above(ax)


def main() -> None:
    """Separate distribution, stress, and chronological backtest views."""
    export_panels("ch22_var_es_loss_distribution", [
        ("historical", "Historical loss sample", draw_histogram),
        ("stress", "Named stress losses", draw_stress),
        ("backtest", "Expanding-window replay backtest", draw_backtest),
        ("tail", "Population tail illustration", draw_tail_schematic),
    ])


if __name__ == "__main__":
    main()
