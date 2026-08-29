"""
Python & AI for Rates, Bonds, and Credit
Chapter 29 · Central-Bank and Macro Document Intelligence

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Policy tone and curve-reaction figure for Chapter 29.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create the policy-tone curve map."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tone = pd.read_csv(DATA / "ch29_policy_tone_outputs.csv")
    curve = pd.read_csv(DATA / "ch29_curve_reaction_snapshot.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(tone["statement_id"], tone["policy_stance"],
                 marker="o", color=BLUE, label="stance")
    axes[0].plot(tone["statement_id"], tone["inflation_concern"],
                 marker="s", color=ORANGE, label="inflation")
    axes[0].set_title("Extracted policy tone", color=NAVY)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(curve["policy_stance"], curve["two_year_change"],
                    s=80, color=LIGHT_BLUE, edgecolor=NAVY)
    axes[1].set_title("Tone and front-end move", color=NAVY)
    axes[1].set_xlabel("Policy stance")
    axes[1].set_ylabel("2Y change (bp)")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch29_policy_tone_curve_map.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
