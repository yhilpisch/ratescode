"""
Python & AI for Rates, Bonds, and Credit
Part VIII · Reconciliation Diagnostics

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Recompute narrow numeric/date comparisons for the frozen teaching records.
These checks do not establish source completeness, units, or release approval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ch28_genai_workflows_fixed_income import load_validation
from ch29_central_bank_macro_document_intelligence import (
    load_tone, load_curve_reactions,
)
from ch31_tool_using_agents_fixed_income_analysis import load_validation_log
from ch32_automated_reporting_communication import reconciliation_table


def numeric_comparison(
    frame: pd.DataFrame,
    source: str="source_value",
    draft: str="draft_value",
) -> pd.DataFrame:
    """Compare finite values exactly; caller must establish matching units."""
    result = frame.copy()
    left = pd.to_numeric(result[source], errors="coerce")
    right = pd.to_numeric(result[draft], errors="coerce")
    match = np.isfinite(left) & np.isfinite(right) & left.eq(right)
    result["computed_status"] = np.where(match, "pass", "fail")
    return result


def extraction_checks() -> pd.DataFrame:
    """Recompute the numeric fields and the separate supplied date check."""
    table = load_validation()
    result = numeric_comparison(table)
    dates = table.field.eq("date")
    left = pd.to_datetime(table.loc[dates, "source_value"], errors="coerce")
    right = pd.to_datetime(table.loc[dates, "draft_value"], errors="coerce")
    match = left.notna() & right.notna() & left.eq(right)
    result.loc[dates, "computed_status"] = np.where(match, "pass", "fail")
    return result


def policy_events() -> pd.DataFrame:
    """Join the synthetic events one-to-one and check duplicated stance data."""
    tone = load_tone()
    curve = load_curve_reactions()
    result = tone.merge(curve, on="date", how="outer", suffixes=("", "_curve"),
                        validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError("each tone event requires one curve observation")
    if not result.policy_stance.eq(result.policy_stance_curve).all():
        raise ValueError("stored stance values disagree across event tables")
    result["date"] = pd.to_datetime(result.date, errors="raise")
    return result.drop(columns=["_merge", "policy_stance_curve"])


def agent_checks() -> pd.DataFrame:
    """Recompute numeric agreement within the claim log, not tool execution."""
    return numeric_comparison(load_validation_log(), source="tool_value")


def report_checks() -> pd.DataFrame:
    """Compare the five supplied metrics without mixing their physical units."""
    return numeric_comparison(reconciliation_table())


def main() -> None:
    """Display recomputed status counts and confirm the event join is usable."""
    for name, table in [("extraction", extraction_checks()),
                         ("agent", agent_checks()),
                         ("report", report_checks())]:
        print(name, table.computed_status.value_counts().to_dict())
    print("joined policy events", len(policy_events()))


if __name__ == "__main__":
    main()
