"""
Python & AI for Rates, Bonds, and Credit
Shared Mathematical Helpers

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from math import erf, sqrt


def normal_cdf(value: float) -> float:
    """Return the standard normal cumulative probability."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))
