"""
Python & AI for Rates, Bonds, and Credit
Chapter 17 · Reduced-Form Credit Models and CDS

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

CDS survival and repricing diagnostics figure for Chapter 17.
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


def recovery_rate() -> float:
    """Load the frozen recovery assumption."""
    table = pd.read_csv(DATA / "ch17_recovery_assumptions.csv")
    return float(table["recovery"].iloc[0])


def cds_inputs(maturity: float | None=None) -> pd.DataFrame:
    """Load discount factors and CDS premium dates."""
    discount = pd.read_csv(DATA / "ch17_discount_curve.csv")
    schedule = pd.read_csv(DATA / "ch17_cds_schedule.csv")
    table = schedule.merge(discount, on="time")
    if maturity is not None:
        table = table.loc[table["time"] <= maturity]
    return table.reset_index(drop=True)


def survival_curve(hazard: float=0.025,
                   maturity: float | None=None) -> pd.DataFrame:
    """Compute survival probabilities for a constant hazard rate."""
    table = cds_inputs(maturity)
    table["survival"] = np.exp(-hazard * table["time"])
    table["default_prob"] = -table["survival"].diff().fillna(
        table["survival"].iloc[0] - 1.0
    )
    table["default_prob"] = table["default_prob"].abs()
    return table


def repricing_table() -> pd.DataFrame:
    """Report hazard rates and repricing residuals for frozen quotes."""
    quotes = pd.read_csv(DATA / "ch17_cds_quotes.csv")
    rows = []
    recovery = recovery_rate()
    for _, row in quotes.iterrows():
        hazard = row["spread_bp"] / 10000.0 / (1.0 - recovery)
        curve = survival_curve(hazard, row["maturity"])
        spread = row["spread_bp"] / 10000.0
        premium = spread * (
            curve["alpha"] * curve["discount"] * curve["survival"]
        ).sum()
        protection = (1.0 - recovery) * (
            curve["discount"] * curve["default_prob"]
        ).sum()
        rows.append(
            {
                "maturity": row["maturity"],
                "residual": premium - protection,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Create the CDS survival and repricing diagnostics figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    curve = survival_curve()
    quotes = repricing_table()
    residuals = quotes["residual"].to_numpy(float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(curve["time"], curve["survival"], marker="o", color=BLUE)
    axes[0].set_title("Survival curve", color=NAVY)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Survival probability")

    axes[1].bar(quotes["maturity"].astype(str), residuals, color=ORANGE)
    axes[1].axhline(0, color=NAVY, lw=1)
    axes[1].set_title("CDS repricing residuals", color=NAVY)
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("PV residual")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch17_cds_survival_repricing_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
