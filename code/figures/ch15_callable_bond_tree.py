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

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_terms() -> dict[str, float]:
    """Load callable bond terms."""
    table = pd.read_csv(DATA / "ch15_callable_bond_terms.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def load_inputs() -> dict[str, float]:
    """Load rate tree inputs."""
    table = pd.read_csv(DATA / "ch15_rate_tree_inputs.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def tree_points() -> tuple[list[float], list[float], list[float]]:
    """Return node coordinates and rates for a short-rate tree."""
    terms = load_terms()
    inputs = load_inputs()
    xs: list[float] = []
    ys: list[float] = []
    rates: list[float] = []
    for step in range(int(terms["maturity_steps"])):
        for up_moves in range(step + 1):
            down_moves = step - up_moves
            rate = inputs["r0"] * inputs["up"] ** up_moves
            rate *= inputs["down"] ** down_moves
            xs.append(float(step))
            ys.append(float(2 * up_moves - step))
            rates.append(rate)
    return xs, ys, rates


def main() -> None:
    """Create the callable bond tree figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    xs, ys, rates = tree_points()
    terms = load_terms()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    colors = [ORANGE if x + 1 >= terms["call_start_step"] else BLUE
              for x in xs]
    ax.scatter(xs, ys, s=360, color=colors, edgecolor=NAVY)
    for x, y, rate in zip(xs, ys, rates, strict=True):
        ax.text(x, y, f"{rate * 100:.2f}%", ha="center", va="center",
                color="white", fontsize=8)
    ax.set_title("Short-rate tree and callable region", color=NAVY)
    ax.set_xlabel("Time step")
    ax.set_yticks([])
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch15_callable_bond_tree.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
