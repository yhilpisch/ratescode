"""
Python & AI for Rates, Bonds, and Credit
Chapter 4 · Yield Curves, Zero Rates, and Forward Rates

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Interpolation and forward-rate diagnostics figure for Chapter 4.
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
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create a figure comparing interpolation choices."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(DATA / "ch04_zero_curve.csv")
    nodes = curve["maturity"].to_numpy(float)
    zeros = curve["zero_rate"].to_numpy(float)
    node_discounts = np.exp(-zeros * nodes)  # node discount factors
    grid = np.linspace(nodes.min(), nodes.max(), 200)

    zero_interp = np.interp(grid, nodes, zeros)  # linear zero rates
    log_discount = np.interp(grid, nodes, np.log(node_discounts))
    discount_interp = np.exp(log_discount)  # log-discount interpolation
    zero_from_log_discount = -np.log(discount_interp) / grid

    zero_discount = np.exp(-zero_interp * grid)
    forward_zero = -np.diff(np.log(zero_discount)) / np.diff(grid)
    forward_log_discount = -np.diff(log_discount) / np.diff(grid)
    forward_grid = 0.5 * (grid[1:] + grid[:-1])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True)

    axes[0].plot(grid, zero_interp * 100, color=BLUE, label="linear zero")
    axes[0].plot(
        grid,
        zero_from_log_discount * 100,
        color=ORANGE,
        label="linear log discount",
    )
    axes[0].scatter(nodes, zeros * 100, color=NAVY, s=22, zorder=3)
    axes[0].set_title("Zero-rate view", color=NAVY)
    axes[0].set_ylabel("Rate in percent")

    axes[1].plot(
        forward_grid,
        forward_zero * 100,
        color=BLUE,
        label="from linear zero",
    )
    axes[1].plot(
        forward_grid,
        forward_log_discount * 100,
        color=ORANGE,
        label="from log discount",
    )
    axes[1].set_title("Forward-rate diagnostic", color=NAVY)

    for ax in axes:
        ax.set_xlabel("Maturity in years")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch04_interpolation_forward_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
