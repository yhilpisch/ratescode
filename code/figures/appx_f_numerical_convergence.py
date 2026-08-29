"""
Python & AI for Rates, Bonds, and Credit
Appendix F · Numerical Methods for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Numerical-convergence figure for Appendix F.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Create a root-finding and Monte Carlo convergence figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    iteration = np.arange(1, 6)
    bracket = np.array([2.00, 1.00, 0.50, 0.25, 0.125])
    paths = np.array([1_000, 10_000, 100_000])
    stderr = np.array([0.090, 0.028, 0.009])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(iteration, bracket, marker="o", color=BLUE, lw=2.0)
    axes[0].set_title("Root bracket shrinks by iteration", color=NAVY)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Bracket width")

    axes[1].plot(paths, stderr, marker="o", color=ORANGE, lw=2.0)
    axes[1].set_xscale("log")
    axes[1].set_title("Monte Carlo error falls slowly", color=NAVY)
    axes[1].set_xlabel("Simulation paths")
    axes[1].set_ylabel("Standard error")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_f_numerical_convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
