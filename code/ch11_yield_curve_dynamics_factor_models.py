"""
Python & AI for Rates, Bonds, and Credit
Chapter 11 · Yield-Curve Dynamics and Factor Models

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
YIELD_COLS = ["y1", "y2", "y5", "y10"]
MATURITIES = np.array([1.0, 2.0, 5.0, 10.0])


def yield_changes() -> pd.DataFrame:
    """Return yield changes in basis points."""
    history = pd.read_csv(DATA / "ch11_yield_curve_history.csv")
    changes = history[YIELD_COLS].diff().dropna() * 10000.0
    return changes.reset_index(drop=True)


def pca_summary() -> pd.DataFrame:
    """Compute principal components of yield changes."""
    changes = yield_changes()
    centered = changes - changes.mean()  # demean factor inputs
    cov = np.cov(centered.to_numpy(float), rowvar=False)  # covariance
    values, vectors = np.linalg.eigh(cov)  # symmetric eigensystem
    order = np.argsort(values)[::-1]  # descending variance order
    values = values[order]
    vectors = vectors[:, order]
    explained = values / values.sum()  # variance shares
    rows = []  # component loading rows
    for idx in range(3):
        rows.append({
            "component": idx + 1,
            "explained": explained[idx],
            "y1": vectors[0, idx],
            "y2": vectors[1, idx],
            "y5": vectors[2, idx],
            "y10": vectors[3, idx],
        })
    return pd.DataFrame(rows)


def nelson_siegel(beta0: float=0.045,
                  beta1: float=-0.015,
                  beta2: float=-0.010,
                  tau: float=2.5) -> pd.DataFrame:
    """Create a Nelson-Siegel curve on the chapter maturity grid."""
    x = MATURITIES / tau
    slope = (1.0 - np.exp(-x)) / x  # slope loading
    curvature = slope - np.exp(-x)  # curvature loading
    fitted = beta0 + beta1 * slope + beta2 * curvature
    return pd.DataFrame({"maturity": MATURITIES, "fitted": fitted})


def main() -> None:
    """Print a compact chapter result summary."""
    print(yield_changes().round(2))
    print(pca_summary().round(4))
    print(nelson_siegel().round(4))


if __name__ == "__main__":
    main()
