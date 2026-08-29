"""
Python & AI for Rates, Bonds, and Credit
Chapter 32 · Automated Reporting and Communication

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Report reconciliation figure for Chapter 32.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create the report reconciliation figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    checks = pd.read_csv(DATA / "ch32_report_validation.csv")
    checks["difference"] = checks["draft_value"] - checks["source_value"]
    colors = [ORANGE if status == "fail" else BLUE
              for status in checks["status"]]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(checks["metric"], checks["difference"], color=colors)
    ax.axhline(0.0, color=NAVY, linewidth=1.0)
    ax.set_title("Draft-to-source reconciliation", color=NAVY)
    ax.set_ylabel("Draft minus source")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ch32_report_reconciliation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
