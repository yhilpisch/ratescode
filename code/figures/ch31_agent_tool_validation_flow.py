"""
Python & AI for Rates, Bonds, and Credit
Chapter 31 · Tool-Using Agents for Fixed-Income Analysis

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Agent tool-validation figure for Chapter 31.
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
LIGHT_BLUE = "#8CB6E8"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    """Create the agent tool-validation flow figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tools = pd.read_csv(DATA / "ch31_tool_outputs.csv")
    log = pd.read_csv(DATA / "ch31_agent_validation_log.csv")
    counts = log["status"].value_counts().reindex(["pass", "fail"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].barh(tools["output_name"], tools["value"], color=BLUE)
    axes[0].set_title("Tool outputs", color=NAVY)
    axes[0].set_xlabel("Value")

    axes[1].bar(counts.index, counts.values, color=[LIGHT_BLUE, ORANGE])
    axes[1].set_title("Claim validation", color=NAVY)
    axes[1].set_ylabel("Checks")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch31_agent_tool_validation_flow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
