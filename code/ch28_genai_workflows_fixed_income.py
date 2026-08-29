"""
Python & AI for Rates, Bonds, and Credit
Chapter 28 · GenAI Workflows for Fixed Income

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
    """Load structured GenAI extractions."""
    return pd.read_csv(DATA / "ch28_structured_extraction.csv")


def load_validation() -> pd.DataFrame:
    """Load the frozen draft validation table."""
    return pd.read_csv(DATA / "ch28_report_validation.csv")


def review_queue(min_confidence: float=0.85) -> pd.DataFrame:
    """Return fields that require human review."""
    extracted = load_extractions()
    mask = (extracted["confidence"] < min_confidence)
    mask = mask | (extracted["needs_review"] == 1)
    columns = ["source_id", "metric", "confidence", "needs_review"]
    return extracted.loc[mask, columns].copy()


def validation_summary() -> pd.DataFrame:
    """Summarize pass/fail checks for the drafted note."""
    checks = load_validation()
    return (checks.groupby("status", as_index=False)
            .size()
            .rename(columns={"size": "checks"}))


def main() -> None:
    """Print a compact chapter result summary."""
    print(review_queue().round(3))
    print(validation_summary())


if __name__ == "__main__":
    main()
