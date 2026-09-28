"""
Python & AI for Rates, Bonds, and Credit
Chapter 22 · VaR, Expected Shortfall, and Stress Testing

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from math import ceil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FACTOR_COLS = ["rate_2y_bp", "rate_5y_bp", "rate_10y_bp", "spread_bp"]


def load_exposures() -> pd.Series:
    """Load frozen DV01 and CS01 exposures by risk factor."""
    table = pd.read_csv(DATA / "ch22_portfolio_exposures.csv")
    return table.set_index("factor")["exposure"]


def historical_losses() -> pd.Series:
    """Convert historical factor changes into positive loss numbers."""
    changes = pd.read_csv(DATA / "ch22_factor_changes.csv")
    dates = pd.to_datetime(changes["date"], errors="raise")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("factor changes must have unique, increasing dates")
    exposures = load_exposures()  # currency loss per bp
    loss = (changes[FACTOR_COLS] * exposures).sum(axis=1)
    loss.index = dates
    return loss.rename("loss")


def var_es(level: float=0.95) -> dict[str, float]:
    """Compute historical VaR and the mean of the worst empirical losses."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    losses = historical_losses()
    var = float(np.quantile(losses, level, method="lower"))
    count = max(1, ceil((1.0 - level) * len(losses) - 1e-12))
    tail = losses.nlargest(count)  # avoid ties at VaR enlarging the tail
    return {"var": var, "es": float(tail.mean())}


def var_backtest(
    level: float=0.95,
    min_observations: int=20,
) -> pd.DataFrame:
    """Forecast VaR from prior losses and test it on each next observation."""
    losses = historical_losses()
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    if min_observations < 2 or len(losses) <= min_observations:
        raise ValueError("need more observations than the initial window")
    rows = []  # each VaR uses only strictly earlier losses
    for index in range(min_observations, len(losses)):
        history = losses.iloc[:index]
        realized = float(losses.iloc[index])
        threshold = float(np.quantile(history, level, method="lower"))
        rows.append({
            "date": losses.index[index],
            "level": level,
            "var": threshold,
            "loss": realized,
            "exception": realized > threshold,
            "training_observations": index,
        })
    return pd.DataFrame(rows)


def parametric_var(level: float=0.95) -> float:
    """Compute normal VaR using the sample mean loss and factor covariance."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    changes = pd.read_csv(DATA / "ch22_factor_changes.csv")
    exposures = load_exposures().reindex(FACTOR_COLS).to_numpy(float)
    cov = changes[FACTOR_COLS].cov().to_numpy(float)  # factor covariance
    sigma = float(np.sqrt(exposures @ cov @ exposures))  # loss volatility
    mean_loss = float(changes[FACTOR_COLS].mean().to_numpy() @ exposures)
    z_value = NormalDist().inv_cdf(level)
    return mean_loss + z_value * sigma


def stress_losses() -> pd.DataFrame:
    """Compute deterministic stress losses from named scenarios."""
    stress = pd.read_csv(DATA / "ch22_stress_scenarios.csv")
    exposures = load_exposures()  # currency loss per bp
    stress["loss"] = (stress[FACTOR_COLS] * exposures).sum(axis=1)
    return stress[["scenario", "loss"]]


def risk_report() -> pd.DataFrame:
    """Build a compact VaR, ES, and stress-test report."""
    hist = var_es()  # historical tail measures
    stress = stress_losses()
    rows = [
        {"metric": "historical VaR 95", "loss": hist["var"]},
        {"metric": "historical ES 95", "loss": hist["es"]},
        {"metric": "parametric VaR 95", "loss": parametric_var()},
        {
            "metric": "worst stress",
            "loss": float(stress["loss"].max()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(risk_report().round(2))
    print(stress_losses().round(2))
    print(var_backtest().head().round(2))


if __name__ == "__main__":
    main()
