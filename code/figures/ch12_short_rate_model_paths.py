"""
Python & AI for Rates, Bonds, and Credit
Chapter 12 · Short-Rate Models: Vasicek, CIR, and Hull-White

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Short-rate model path figure for Chapter 12.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
import sys


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from ch12_short_rate_models import simulate_paths


def simulate(model: str, steps: int=24) -> np.ndarray:
    """Return a path from the chapter's canonical model implementation."""
    return simulate_paths(model, steps=steps, paths=1)[:, 0]


def main() -> None:
    """Create the short-rate path figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    months = np.arange(25)
    colors = {"vasicek": BLUE, "cir": ORANGE, "hull-white": LIGHT_BLUE}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for model, color in colors.items():
        ax.plot(months, simulate(model) * 100, color=color,
                label=model.title())
    ax.set_title("Short-rate model paths", color=NAVY)
    ax.set_xlabel("Month")
    ax.set_ylabel("Short rate percent")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch12_short_rate_model_paths.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
