"""
Python & AI for Rates, Bonds, and Credit
Appendix G · Optimization Refresher

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Feasible-set figure for Appendix G.
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
    """Create a constrained-optimization figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, 200)
    upper = np.minimum(1.0 - x, 0.75)
    objective = 0.65 - 0.35 * x

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.fill_between(x, 0.0, upper, color=BLUE, alpha=0.18,
                    label="feasible region")
    ax.plot(x, upper, color=BLUE, lw=2.0)
    ax.plot(x, objective, color=ORANGE, lw=2.0, label="objective contour")
    ax.scatter([0.35], [0.40], color=NAVY, s=45, zorder=3, label="solution")
    ax.set_title("Constraints define the feasible set", color=NAVY)
    ax.set_xlabel("Weight x1")
    ax.set_ylabel("Weight x2")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_g_optimization_feasible_set.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
