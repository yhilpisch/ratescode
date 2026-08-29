"""
Python & AI for Rates, Bonds, and Credit
Chapter 30 · Credit Documents, Covenants, and Research Notes

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


def load_extractions() -> pd.DataFrame:
    """Load covenant extractions from synthetic credit memos."""
    return pd.read_csv(DATA / "ch30_covenant_extractions.csv")


def covenant_headroom() -> pd.DataFrame:
    """Compute covenant headroom with metric-specific direction."""
    cov = load_extractions()
    is_leverage = cov["metric"].eq("net_leverage")
    cov["headroom"] = cov["threshold"] - cov["extracted_value"]
    cov.loc[~is_leverage, "headroom"] = (
        cov.loc[~is_leverage, "extracted_value"]
        - cov.loc[~is_leverage, "threshold"]
    )
    columns = ["issuer", "metric", "extracted_value", "threshold",
               "headroom", "status"]
    return cov[columns].copy()


def review_table(min_headroom: float=0.40) -> pd.DataFrame:
    """Flag issuers where covenant headroom is thin."""
    table = covenant_headroom()
    table["review"] = table["headroom"].lt(min_headroom)
    return table[["issuer", "headroom", "review"]].copy()


def main() -> None:
    """Print a compact chapter result summary."""
    print(covenant_headroom().round(2))
    print(review_table().round(2))


if __name__ == "__main__":
    main()
