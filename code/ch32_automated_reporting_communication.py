"""
Python & AI for Rates, Bonds, and Credit
Chapter 32 · Automated Reporting and Communication

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


def load_validation() -> pd.DataFrame:
    """Load report-to-source reconciliation checks."""
    return pd.read_csv(DATA / "ch32_report_validation.csv")


def reconciliation_table() -> pd.DataFrame:
    """Compute draft errors relative to verified source metrics."""
    checks = load_validation()
    checks["difference"] = checks["draft_value"] - checks["source_value"]
    columns = ["metric", "source_value", "draft_value",
               "difference", "unit", "status"]
    return checks[columns].copy()


def failed_checks() -> pd.DataFrame:
    """Return report sentences that must be revised."""
    table = reconciliation_table()
    return table.loc[table["status"].eq("fail")].copy()


def main() -> None:
    """Print a compact chapter result summary."""
    print(reconciliation_table().round(2))
    print(failed_checks().round(2))


if __name__ == "__main__":
    main()
