"""
Python & AI for Rates, Bonds, and Credit
Chapter 17 · Reduced-Form Credit Models and CDS

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def recovery_rate() -> float:
    """Load the frozen recovery assumption."""
    table = pd.read_csv(DATA / "ch17_recovery_assumptions.csv")
    return float(table["recovery"].iloc[0])


def cds_inputs() -> pd.DataFrame:
    """Load discount factors and CDS premium dates."""
    discount = pd.read_csv(DATA / "ch17_discount_curve.csv")
    schedule = pd.read_csv(DATA / "ch17_cds_schedule.csv")
    return schedule.merge(discount, on="time")


def survival_curve(hazard: float=0.025) -> pd.DataFrame:
    """Compute survival probabilities for a constant hazard rate."""
    table = cds_inputs()
    table["survival"] = np.exp(-hazard * table["time"])  # survival prob
    table["default_prob"] = -table["survival"].diff().fillna(
        table["survival"].iloc[0] - 1.0
    )
    table["default_prob"] = table["default_prob"].abs()
    return table


def cds_legs(spread_bp: float=150.0,
             hazard: float=0.025) -> dict[str, float]:
    """Compute approximate CDS premium and protection legs."""
    curve = survival_curve(hazard)
    spread = spread_bp / 10000.0  # decimal spread
    premium = spread * (curve["alpha"] * curve["discount"]
                        * curve["survival"]).sum()
    protection = (1.0 - recovery_rate()) * (
        curve["discount"] * curve["default_prob"]
    ).sum()
    return {"premium_leg": float(premium), "protection_leg": float(protection)}


def implied_hazard(spread_bp: float=150.0) -> float:
    """Approximate the flat hazard rate from spread and recovery.

    Uses the standard simple formula spread/(1-recovery). A flat hazard
    will generally not reprice a non-flat market spread curve exactly,
    so repricing residuals are expected and reported as diagnostics.
    """
    return spread_bp / 10000.0 / (1.0 - recovery_rate())


def repricing_table() -> pd.DataFrame:
    """Report hazard rates and repricing residuals for frozen quotes."""
    quotes = pd.read_csv(DATA / "ch17_cds_quotes.csv")
    rows = []  # calibration diagnostics
    for _, row in quotes.iterrows():
        hazard = implied_hazard(row["spread_bp"])
        legs = cds_legs(row["spread_bp"], hazard)
        residual = legs["premium_leg"] - legs["protection_leg"]
        rows.append({
            "maturity": row["maturity"],
            "spread_bp": row["spread_bp"],
            "hazard": hazard,
            "residual": residual,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(survival_curve().head().round(6))
    print(repricing_table().round(6))


if __name__ == "__main__":
    main()
