"""
Python & AI for Rates, Bonds, and Credit
Chapter 17 · Reduced-Form Credit Models and CDS

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

CDS survival and repricing diagnostics figure for Chapter 17.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from credit_figure_layout import (
    NAVY, BLUE, ORANGE, export_panels, legend_above,
)

from ch17_reduced_form_credit_models_cds import (
    survival_curve, repricing_table,
)


def draw_survival(ax) -> None:
    """Show one flat 2.5%-per-year hazard, including survival at zero."""
    curve = survival_curve()
    time = np.r_[0, curve.time]
    survival = np.r_[1, curve.survival]
    ax.plot(time, survival * 100, marker="o", color=BLUE, lw=2)
    ax.set(xlabel="Horizon (years)", ylabel="Risk-neutral survival (%)",
           ylim=(94, 100.5))


def draw_residuals(ax) -> None:
    """Scale premium-minus-protection residuals to bp of notional."""
    table = repricing_table()
    bars = ax.bar([f"{t:g}Y" for t in table.maturity],
                  table.residual * 10000, color=ORANGE, width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=NAVY)
    ax.axhline(0, color=NAVY, lw=1)
    ax.set(xlabel="Quote maturity; independent flat-hazard shortcut",
           ylabel="PV residual (bp of notional)", ylim=(-1.5, 0.15))


def main() -> None:
    """Separate a survival illustration from independent quote checks."""
    export_panels("ch17_cds_survival_repricing_diagnostics", [
        ("survival", "Flat-hazard survival", draw_survival),
        ("residuals", "Independent quote residuals", draw_residuals),
    ])


if __name__ == "__main__":
    main()
