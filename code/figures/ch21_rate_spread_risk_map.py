"""
Python & AI for Rates, Bonds, and Credit
Chapter 21 · Interest-Rate and Spread Risk

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Rate and spread risk map figure for Chapter 21.
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exposure and scenario data."""
    portfolio = pd.read_csv(DATA / "ch21_risk_portfolio.csv")
    scenarios = pd.read_csv(DATA / "ch21_shock_scenarios.csv")
    return portfolio, scenarios


def scenario_losses(portfolio: pd.DataFrame,
                    scenarios: pd.DataFrame) -> pd.Series:
    """Compute scenario losses from rate DV01 and CS01 exposures."""
    dv01_cols = ["dv01_2y", "dv01_5y", "dv01_10y"]
    shock_cols = ["rate_2y_bp", "rate_5y_bp", "rate_10y_bp"]
    exposures = portfolio[dv01_cols + ["cs01"]].sum()  # total exposure
    losses = []  # scenario loss values
    for _, row in scenarios.iterrows():
        rate_loss = sum(
            exposures[dv] * row[shock]
            for dv, shock in zip(dv01_cols, shock_cols, strict=True)
        )
        losses.append(rate_loss + exposures["cs01"] * row["spread_bp"])
    return pd.Series(losses, index=scenarios["scenario"])


def main() -> None:
    """Create the rate and spread risk map figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    portfolio, scenarios = load_data()
    labels = ["2Y", "5Y", "10Y", "CS01"]  # risk columns
    matrix = portfolio[["dv01_2y", "dv01_5y", "dv01_10y", "cs01"]]
    matrix = matrix.to_numpy(float)
    losses = scenario_losses(portfolio, scenarios)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    image = axes[0].imshow(matrix, cmap="Blues", aspect="auto")
    axes[0].set_title("Instrument exposures", color=NAVY)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticks(np.arange(len(portfolio)))
    axes[0].set_yticklabels(portfolio["instrument"], fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].barh(losses.index, losses.values, color=ORANGE)
    axes[1].set_title("Scenario loss", color=NAVY)
    axes[1].set_xlabel("Currency units")
    axes[1].invert_yaxis()

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch21_rate_spread_risk_map.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
