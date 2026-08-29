"""
Python & AI for Rates, Bonds, and Credit
Appendix B · Itô Calculus Refresher

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Itô correction figure for Appendix B.
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
    """Create a figure for the Itô log-process correction."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    sigma = np.linspace(0.05, 0.40, 60)
    correction = -0.5 * sigma**2
    base_drift = np.full_like(sigma, 0.03)
    log_drift = base_drift + correction

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(sigma * 100, base_drift * 100, color=BLUE, lw=2.0,
            label="level-process drift")
    ax.plot(sigma * 100, log_drift * 100, color=ORANGE, lw=2.0,
            label="log-process drift")
    ax.fill_between(
        sigma * 100,
        log_drift * 100,
        base_drift * 100,
        color=ORANGE,
        alpha=0.15,
        label="Itô correction",
    )
    ax.set_title("The Itô correction lowers log drift", color=NAVY)
    ax.set_xlabel("Volatility in percent")
    ax.set_ylabel("Annualised drift in percent")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_b_ito_correction.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
