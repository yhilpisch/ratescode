"""
Python & AI for Rates, Bonds, and Credit
Chapter 29 · Central-Bank and Macro Document Intelligence

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_tone() -> pd.DataFrame:
    """Load extracted policy-tone fields."""
    return pd.read_csv(DATA / "ch29_policy_tone_outputs.csv")


def load_curve_reactions() -> pd.DataFrame:
    """Load curve moves around policy statements."""
    return pd.read_csv(DATA / "ch29_curve_reaction_snapshot.csv")


def tone_table() -> pd.DataFrame:
    """Return the main policy-tone diagnostics."""
    tone = load_tone()
    columns = ["statement_id", "policy_stance",
               "inflation_concern", "growth_concern", "confidence"]
    return tone[columns].copy()


def reaction_link() -> pd.DataFrame:
    """Link extracted stance to curve reactions without causal claims."""
    curve = load_curve_reactions()
    curve["front_end_per_stance"] = (
        curve["two_year_change"] / curve["policy_stance"]
    )
    columns = ["date", "policy_stance", "two_year_change",
               "ten_year_change", "curve_slope_change",
               "front_end_per_stance"]
    return curve[columns].copy()


def main() -> None:
    """Print a compact chapter result summary."""
    print(tone_table().round(2))
    print(reaction_link().round(2))


if __name__ == "__main__":
    main()
