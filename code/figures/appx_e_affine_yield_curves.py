"""
Python & AI for Rates, Bonds, and Credit
Appendix E · Affine Term-Structure Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Affine yield-curve comparison figure for Appendix E.
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
    """Create an affine-model yield-curve comparison figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    tau = np.linspace(0.5, 15.0, 120)
    vasicek = 2.8 + 1.4 * (1.0 - np.exp(-0.22 * tau))
    cir = 3.0 + 1.2 * (1.0 - np.exp(-0.35 * tau))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(tau, vasicek, color=BLUE, lw=2.1, label="Vasicek-style")
    ax.plot(tau, cir, color=ORANGE, lw=2.1, label="CIR-style")
    ax.set_title("Affine models produce smooth term structures", color=NAVY)
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel("Zero rate in percent")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_e_affine_yield_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
