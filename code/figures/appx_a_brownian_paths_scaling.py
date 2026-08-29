"""
Python & AI for Rates, Bonds, and Credit
Appendix A · Stochastic Processes and Brownian Motion

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Brownian paths and scaling figure for Appendix A.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np


NAVY = "#001F5B"
BLUE = "#2F6DB5"
LIGHT_BLUE = "#8CB6E8"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Create a Brownian-path and scaling figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1300)
    dt = 1.0 / 120.0
    times = np.linspace(0.0, 1.0, 121)
    paths = []
    for _ in range(4):
        increments = np.sqrt(dt) * rng.standard_normal(120)
        paths.append(np.r_[0.0, np.cumsum(increments)])
    horizons = np.array([1 / 12, 0.25, 0.5, 1.0])
    scale = np.sqrt(horizons)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    colors = [LIGHT_BLUE, BLUE, ORANGE, NAVY]
    for path, color in zip(paths, colors, strict=True):
        axes[0].plot(times, path, color=color, lw=1.6)
    axes[0].set_title("Brownian paths over one year", color=NAVY)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Path level")

    axes[1].plot(horizons, scale, marker="o", color=BLUE, lw=2.0)
    axes[1].set_title("Volatility scales with sqrt(time)", color=NAVY)
    axes[1].set_xlabel("Horizon in years")
    axes[1].set_ylabel("Scaling factor")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_a_brownian_paths_scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
