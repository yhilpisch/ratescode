"""
Python & AI for Rates, Bonds, and Credit
Chapter 2 · Cash Flows, Discounting, and Present Value

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Present-value contribution sensitivity figure for Chapter 2.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
LIGHT_BLUE = "#8CB6E8"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def pv_contributions(rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, cash flows, and present-value contributions."""
    table = pd.read_csv(DATA / "ch02_cashflows.csv")
    times = table["time"].to_numpy(float)
    cashflows = table["cashflow"].to_numpy(float)
    discounts = np.exp(-rate * times)  # continuous-compounding discounts
    return times, cashflows, cashflows * discounts


def main() -> None:
    """Create the present-value contribution sensitivity figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("lower rate", 0.03, LIGHT_BLUE),
        ("base rate", 0.04, BLUE),
        ("higher rate", 0.05, ORANGE),
    ]
    width = 0.22  # grouped-bar width
    offsets = np.array([-width, 0.0, width])  # bar offsets by scenario

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for offset, (label, rate, color) in zip(offsets, scenarios):
        times, _, parts = pv_contributions(rate)
        ax.bar(times + offset, parts, width=width, label=label, color=color)

    ax.set_title("Present-value contribution by payment date", color=NAVY)
    ax.set_xlabel("Payment time in years")
    ax.set_ylabel("Present-value contribution")
    ax.set_xticks(times)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch02_pv_contribution_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
