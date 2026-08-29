"""
Python & AI for Rates, Bonds, and Credit
Chapter 1 · The Fixed-Income Landscape and Problem Map

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations


def chapter_summary() -> dict[str, str]:
    """Return a compact summary for the opening chapter."""
    # Keep the opening scaffold lightweight and fully reproducible.
    return {
        "book": "Python & AI for Rates, Bonds, and Credit",
        "chapter": "The Fixed-Income Landscape and Problem Map",
        "focus": "rates, bonds, credit, and reproducible workflows",
    }


if __name__ == "__main__":
    # Print the summary so validation can execute the script safely.
    for key, value in chapter_summary().items():
        print(f"{key}: {value}")
