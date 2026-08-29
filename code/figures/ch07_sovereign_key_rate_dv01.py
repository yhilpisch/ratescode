"""
Python & AI for Rates, Bonds, and Credit
Chapter 7 · Government Bonds and Sovereign Curve Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Sovereign key-rate DV01 figure for Chapter 7.
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


def price_bond(
    coupon: float,
    maturity: int,
    curve_times: np.ndarray,
    zero_rates: np.ndarray,
) -> float:
    """Price a fixed-rate bond from node zero rates."""
    pay_times = np.arange(1.0, maturity + 1.0)  # annual payment dates
    cashflows = np.full(pay_times.size, coupon * 100.0)  # coupons
    cashflows[-1] += 100.0  # final principal
    zeros = np.interp(pay_times, curve_times, zero_rates)  # cash-flow zeros
    discounts = np.exp(-zeros * pay_times)  # discount factors
    return float(np.sum(cashflows * discounts))


def key_rate_dv01(
    coupon: float,
    maturity: int,
    curve_times: np.ndarray,
    zero_rates: np.ndarray,
) -> np.ndarray:
    """Compute key-rate DV01 values by shocking each curve node."""
    bp = 0.0001  # one basis point
    values = []  # key-rate results
    for idx in range(curve_times.size):
        down = zero_rates.copy()  # lower node shock
        up = zero_rates.copy()  # higher node shock
        down[idx] -= bp
        up[idx] += bp
        dv01 = (
            price_bond(coupon, maturity, curve_times, down)
            - price_bond(coupon, maturity, curve_times, up)
        ) / 2.0
        values.append(dv01)
    return np.array(values)


def main() -> None:
    """Create the sovereign key-rate DV01 figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(DATA / "ch07_sovereign_curve.csv")
    bond_table = pd.read_csv(DATA / "ch07_sovereign_bonds.csv")
    curve_times = curve["maturity"].to_numpy(float)
    zero_rates = curve["zero_rate"].to_numpy(float)
    bonds = [
        (row.bond, float(row.coupon), int(row.maturity))
        for row in bond_table.itertuples()
    ]
    colors = [LIGHT_BLUE, BLUE, ORANGE]  # bond colors

    x = np.arange(curve_times.size)  # node locations
    width = 0.24  # grouped-bar width

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    for offset, (bond, coupon, maturity), color in zip(
        [-width, 0.0, width],
        bonds,
        colors,
    ):
        values = key_rate_dv01(coupon, maturity, curve_times, zero_rates)
        ax.bar(x + offset, values, width=width, label=bond, color=color)

    ax.set_title("Sovereign bonds load on different curve nodes", color=NAVY)
    ax.set_xlabel("Curve node in years")
    ax.set_ylabel("Key-rate DV01 per 100 face")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}Y" for t in curve_times])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch07_sovereign_key_rate_dv01.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
