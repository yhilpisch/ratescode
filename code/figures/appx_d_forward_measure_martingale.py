"""
Python & AI for Rates, Bonds, and Credit
Appendix D · Change of Numeraire and Forward Measures

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Forward-measure martingale figure for Appendix D.
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
    """Create a forward-measure martingale illustration."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    times = np.arange(5)
    paths = np.array(
        [
            [0.040, 0.041, 0.039, 0.040, 0.040],
            [0.040, 0.039, 0.041, 0.040, 0.040],
            [0.040, 0.040, 0.040, 0.041, 0.040],
        ]
    )
    mean_path = paths.mean(axis=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = [LIGHT_BLUE, BLUE, ORANGE]
    for path, color in zip(paths, colors, strict=True):
        ax.plot(times, path * 100, color=color, lw=1.6, alpha=0.9)
    ax.plot(times, mean_path * 100, color=NAVY, lw=2.4, label="path average")
    ax.axhline(4.0, color=NAVY, ls="--", lw=1.0, label="initial forward")
    ax.set_title("Under a forward measure, the mean stays anchored",
                 color=NAVY)
    ax.set_xlabel("Observation step")
    ax.set_ylabel("Forward rate in percent")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "appx_d_forward_measure_martingale.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
