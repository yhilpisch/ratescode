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
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax, ax_change) = plt.subplots(
        2, 1, figsize=(7.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    times, _, base_parts = pv_contributions(0.04)
    spacing = float(np.diff(times).min())
    width = min(0.09, spacing / 8)
    offsets = width * np.array([-1.25, 0.0, 1.25])
    handles = []

    for offset, (label, rate, color) in zip(offsets, scenarios):
        times, _, parts = pv_contributions(rate)
        bars = ax.bar(
            times + offset, parts, width=width, label=label, color=color
        )
        handles.append(bars[0])
        change_pct = (parts / base_parts - 1.0) * 100
        ax_change.bar(times + offset, change_pct, width=width, color=color)

    ax.set_title("Cash-flow value and relative rate-shock effect", color=NAVY)
    ax.set_ylabel("PV contribution ($)")
    ax_change.set_ylabel("Change vs base (%)")
    ax_change.set_xlabel("Payment time in years")
    ax.set_xticks(times)
    fig.legend(
        handles,
        [label for label, _, _ in scenarios],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    ax_change.axhline(0.0, color=NAVY, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax_change.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax_change.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax_change.xaxis.label.set_color(NAVY)
    ax_change.yaxis.label.set_color(NAVY)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "ch02_pv_contribution_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
