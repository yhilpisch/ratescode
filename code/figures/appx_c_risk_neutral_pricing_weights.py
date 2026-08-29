"""
Python & AI for Rates, Bonds, and Credit
Appendix C · Risk-Neutral Valuation

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Risk-neutral pricing-weights figure for Appendix C.
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
    """Create a pricing-weights figure for a two-state payoff."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    states = ["down state", "up state"]
    payoff = np.array([2.0, 8.0])
    real_world = np.array([0.65, 0.35])
    risk_neutral = np.array([0.40, 0.60])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(states, payoff, color=BLUE)
    axes[0].set_title("State payoffs", color=NAVY)
    axes[0].set_ylabel("Payoff")

    width = 0.34
    x = np.arange(len(states))
    axes[1].bar(x - width / 2, real_world, width=width,
                color=ORANGE, label="real-world")
    axes[1].bar(x + width / 2, risk_neutral, width=width,
                color=BLUE, label="risk-neutral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(states)
    axes[1].set_title("Pricing weights differ from forecasts", color=NAVY)
    axes[1].set_ylabel("Probability weight")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_c_risk_neutral_pricing_weights.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
