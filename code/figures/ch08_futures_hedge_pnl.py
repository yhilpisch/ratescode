"""
Python & AI for Rates, Bonds, and Credit
Chapter 8 · Bond Futures, CTD Bonds, and Duration Hedging

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Bond futures hedge profit/loss figure for Chapter 8.
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
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def hedge_inputs() -> tuple[float, float, int]:
    """Return portfolio DV01, futures DV01, and rounded hedge count."""
    hedge = pd.read_csv(DATA / "ch08_futures_hedge_inputs.csv")
    contract = pd.read_csv(DATA / "ch08_futures_contract.csv")
    values = hedge["value"].to_numpy(float)
    durations = hedge["modified_duration"].to_numpy(float)
    portfolio_dv01 = float(np.sum(values * durations * 0.0001))
    ctd_dv01 = float(
        contract.loc[contract["field"] == "ctd_dv01", "value"].iloc[0]
    )
    factor = float(
        contract.loc[
            contract["field"] == "conversion_factor", "value"
        ].iloc[0]
    )
    futures_dv01 = ctd_dv01 / factor
    contracts = round(-portfolio_dv01 / futures_dv01)  # hedge contracts
    return portfolio_dv01, futures_dv01, contracts


def main() -> None:
    """Create the futures hedge scenario figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    contract = pd.read_csv(DATA / "ch08_futures_contract.csv")
    portfolio_dv01, futures_dv01, contracts = hedge_inputs()
    base_shock = float(
        contract.loc[contract["field"] == "shock_bp", "value"].iloc[0]
    )
    shocks = np.array([-2, -1, 0, 1, 2], dtype=float) * base_shock
    cash_pnl = -portfolio_dv01 * shocks  # unhedged cash result
    futures_pnl = -contracts * futures_dv01 * shocks  # hedge result
    hedged_pnl = cash_pnl + futures_pnl  # residual result

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True)
    axes[0].plot(
        shocks,
        cash_pnl / 1000,
        marker="o",
        color=BLUE,
        label="cash",
    )
    axes[0].axhline(0.0, color=NAVY, lw=1.0)
    axes[0].set_title("Unhedged cash portfolio", color=NAVY)
    axes[0].set_ylabel("Profit/loss in thousands")

    axes[1].plot(
        shocks,
        hedged_pnl / 1000,
        marker="o",
        color=ORANGE,
        label="cash plus futures",
    )
    axes[1].axhline(0.0, color=NAVY, lw=1.0)
    axes[1].set_title("After futures overlay", color=NAVY)
    axes[1].set_ylabel("Residual in thousands")

    for ax in axes:
        ax.set_xlabel("Yield shock in basis points")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)

    fig.tight_layout()
    fig.savefig(out / "ch08_futures_hedge_pnl.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
