"""
Python & AI for Rates, Bonds, and Credit
Chapter 24 · Machine Learning Workflow for Fixed Income

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Machine learning workflow and validation figure for Chapter 24.
"""
from __future__ import annotations

import os
import sys
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
sys.path.insert(0, str(ROOT / "code"))

from ch24_machine_learning_workflow_fixed_income import time_split


def main() -> None:
    """Create the ML workflow validation figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(DATA / "ch24_ml_feature_panel.csv")
    metrics = pd.read_csv(DATA / "ch24_model_comparison.csv")
    train, test = time_split()
    train_n = len(train)
    test_n = len(test)
    split = len(panel) - test_n
    purged_n = split - train_n

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].barh(["observations"], [train_n], color=BLUE, label="train")
    axes[0].barh(
        ["observations"], [purged_n], left=[train_n],
        color=LIGHT_BLUE, label="purged",
    )
    axes[0].barh(
        ["observations"], [test_n], left=[split],
        color=ORANGE, label="test",
    )
    axes[0].set_title("Purged chronological split", color=NAVY)
    axes[0].set_xlabel("Observation index")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(metrics["model"], metrics["mae"], color=LIGHT_BLUE)
    axes[1].set_title("Illustrative prediction errors", color=NAVY)
    axes[1].set_ylabel("MAE")

    for ax in axes:
        ax.tick_params(colors=NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "ch24_ml_workflow_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
