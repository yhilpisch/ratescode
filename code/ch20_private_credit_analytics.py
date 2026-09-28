"""
Python & AI for Rates, Bonds, and Credit
Chapter 20 · Private Credit Analytics

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


def loan_book() -> pd.DataFrame:
    """Load loans and borrower covenant metrics."""
    loans = pd.read_csv(DATA / "ch20_private_credit_loans.csv")
    metrics = pd.read_csv(DATA / "ch20_borrower_metrics.csv")
    return loans.merge(metrics, on="borrower")


def borrower_metrics() -> pd.DataFrame:
    """Compute leverage, coverage, and covenant headroom."""
    table = loan_book()
    table["coupon_rate"] = table["base_rate"] + table["spread"]
    table["coupon_income"] = table["principal"] * table["coupon_rate"]
    table["leverage"] = table["debt"] / table["ebitda"]
    table["coverage"] = table["ebitda"] / table["interest"]
    table["leverage_headroom"] = table["max_leverage"] - table["leverage"]
    table["coverage_headroom"] = table["coverage"] - table["min_coverage"]
    cols = ["borrower", "coupon_income", "leverage",
            "coverage", "leverage_headroom", "coverage_headroom"]
    return table[cols]


def stress_table() -> pd.DataFrame:
    """Aggregate borrower-level stress results by scenario."""
    details = borrower_stress()
    return details.groupby("scenario", sort=False).agg(
        min_coverage=("coverage", "min"),
        income=("coupon_income", "sum"),
        leverage_breaches=("leverage_breach", "sum"),
        coverage_breaches=("coverage_breach", "sum"),
    ).reset_index()


def borrower_stress() -> pd.DataFrame:
    """Report each borrower's covenant status for the frozen scenarios.

    Only the sampled floating loan principal passes a base-rate shock through
    to total borrower interest; all other interest is held fixed.
    """
    book = loan_book()
    scenarios = pd.read_csv(DATA / "ch20_private_credit_stress_scenarios.csv")
    rows = []  # stress diagnostics
    for _, scenario in scenarios.iterrows():
        stressed_rate = book["base_rate"] + scenario["base_rate_shock"]
        income = book["principal"] * (stressed_rate + book["spread"])
        interest = (
            book["interest"] + book["principal"] * scenario["base_rate_shock"]
        )
        ebitda = book["ebitda"] * (1.0 + scenario["ebitda_shock"])
        coverage = ebitda / interest
        leverage = book["debt"] / ebitda
        for idx in book.index:
            rows.append({
                "scenario": scenario["scenario"],
                "borrower": book.loc[idx, "borrower"],
                "coupon_income": float(income.loc[idx]),
                "leverage": float(leverage.loc[idx]),
                "coverage": float(coverage.loc[idx]),
                "leverage_headroom": float(
                    book.loc[idx, "max_leverage"] - leverage.loc[idx]
                ),
                "coverage_headroom": float(
                    coverage.loc[idx] - book.loc[idx, "min_coverage"]
                ),
            })
    detail = pd.DataFrame(rows)
    detail["leverage_breach"] = detail["leverage_headroom"] < 0
    detail["coverage_breach"] = detail["coverage_headroom"] < 0
    return detail
    return pd.DataFrame(rows)


def covenant_snippets() -> pd.DataFrame:
    """Load frozen covenant text snippets."""
    return pd.read_csv(DATA / "ch20_covenant_snippets.csv")


def main() -> None:
    """Print a compact chapter result summary."""
    print(borrower_metrics().round(3))
    print(borrower_stress().round(3))
    print(stress_table().round(3))
    print(covenant_snippets()[["borrower"]])


if __name__ == "__main__":
    main()
