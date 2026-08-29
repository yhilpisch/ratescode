"""
Python & AI for Rates, Bonds, and Credit
Chapter 28 · GenAI Workflows for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

GenAI extraction and validation figure for Chapter 28.
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
    """Create the GenAI workflow validation figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    extracts = pd.read_csv(DATA / "ch28_structured_extraction.csv")
    checks = pd.read_csv(DATA / "ch28_report_validation.csv")
    counts = checks["status"].value_counts().reindex(["pass", "fail"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].barh(extracts["metric"], extracts["confidence"], color=BLUE)
    axes[0].axvline(0.85, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[0].set_title("Extraction confidence", color=NAVY)
    axes[0].set_xlabel("Confidence")

    axes[1].bar(counts.index, counts.values, color=[LIGHT_BLUE, ORANGE])
    axes[1].set_title("Draft validation checks", color=NAVY)
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch28_genai_workflow_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
