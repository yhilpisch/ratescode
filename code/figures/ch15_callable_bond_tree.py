"""
Python & AI for Rates, Bonds, and Credit
Chapter 15 · Callable Bonds and Embedded Options

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Callable bond tree figure for Chapter 15.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from ch15_callable_bonds_embedded_options import load_terms, rate_tree

NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"


def draw_tree(tree: list[list[float]], filename: str) -> None:
    """Draw connected nodes, legible labels, and call eligibility."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    terms = load_terms()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.8, 3.0))
    ax.axvspan(terms["call_start_step"] - 0.35, len(tree) - 0.6,
               color=ORANGE, alpha=0.12)
    for step, level in enumerate(tree):
        for node, rate in enumerate(level):
            y = 2 * node - step
            if step < len(tree) - 1:
                for target in [y - 1, y + 1]:
                    ax.plot([step, step + 1], [y, target],
                            color="#A9B6C8", lw=1.3, zorder=1)
            color = ORANGE if step >= terms["call_start_step"] else BLUE
            ax.text(step, y, f"{rate:.2%}", ha="center", va="center",
                    color=NAVY, fontsize=14, zorder=3,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                          "edgecolor": color, "linewidth": 1.7})
    ax.text(0.5, 3.65, "Call protection", ha="center", color=BLUE)
    ax.text(2.5, 3.65, "Call eligible at par", ha="center", color=ORANGE)
    ax.set(xlim=(-0.3, 3.4), ylim=(-3.6, 4.0),
           xlabel="Time (years); final coupon + principal paid at year 4")
    ax.set_xticks(range(4))
    ax.set_yticks([])
    ax.grid(False)
    ax.tick_params(colors=NAVY)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)


def main() -> None:
    """Create the book tree and the deck's explicitly rounded example."""
    draw_tree(rate_tree(), "ch15_callable_bond_tree.pdf")
    rounded = [[0.0350], [0.0315, 0.0402],
               [0.0284, 0.0362, 0.0463],
               [0.0255, 0.0326, 0.0417, 0.0532]]
    draw_tree(rounded, "ch15_callable_bond_tree_slides.pdf")


if __name__ == "__main__":
    main()
