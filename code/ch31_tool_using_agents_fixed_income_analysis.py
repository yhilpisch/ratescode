"""
Python & AI for Rates, Bonds, and Credit
Chapter 31 · Tool-Using Agents for Fixed-Income Analysis

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


def load_validation_log() -> pd.DataFrame:
    """Load agent claims and tool-verified values."""
    return pd.read_csv(DATA / "ch31_agent_validation_log.csv")


def failed_claims() -> pd.DataFrame:
    """Return agent claims that fail tool reconciliation."""
    log = load_validation_log()
    columns = ["step", "claim", "tool_value", "draft_value", "status"]
    return log.loc[log["status"].eq("fail"), columns].copy()


def tool_audit() -> pd.DataFrame:
    """Count pass/fail outcomes in the agent validation log."""
    log = load_validation_log()
    return (log.groupby("status", as_index=False)
            .size()
            .rename(columns={"size": "checks"}))


def main() -> None:
    """Print a compact chapter result summary."""
    print(failed_claims().round(2))
    print(tool_audit())


if __name__ == "__main__":
    main()
