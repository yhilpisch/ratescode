"""
Python & AI for Rates, Bonds, and Credit
Chapter 15 · Callable Bonds and Embedded Options

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_terms() -> dict[str, float]:
    """Load frozen callable bond terms."""
    table = pd.read_csv(DATA / "ch15_callable_bond_terms.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def load_tree_inputs() -> dict[str, float]:
    """Load frozen short-rate tree inputs."""
    table = pd.read_csv(DATA / "ch15_rate_tree_inputs.csv")
    return dict(zip(table["field"], table["value"], strict=True))


def rate_tree(r0: float | None=None) -> list[list[float]]:
    """Build a recombining short-rate tree."""
    terms = load_terms()
    inputs = load_tree_inputs()
    steps = int(terms["maturity_steps"])
    start_rate = inputs["r0"] if r0 is None else r0
    tree: list[list[float]] = []  # rate levels by time step
    for step in range(steps):
        rates = []  # rates at this step
        for up_moves in range(step + 1):
            down_moves = step - up_moves
            rate = start_rate * inputs["up"] ** up_moves
            rate *= inputs["down"] ** down_moves
            rates.append(rate)
        tree.append(rates)
    return tree


def bond_value(callable_bond: bool=True,
               r0: float | None=None) -> float:
    """Value callable or non-callable bond by backward induction."""
    terms = load_terms()
    inputs = load_tree_inputs()
    steps = int(terms["maturity_steps"])
    coupon = terms["coupon"] * terms["notional"]
    call_price = terms["call_price"]
    call_start = int(terms["call_start_step"])
    values = [terms["notional"]] * (steps + 1)
    tree = rate_tree(r0)
    for step in range(steps - 1, -1, -1):
        new_values = []  # node values one step earlier
        for node, rate in enumerate(tree[step]):
            expected = 0.5 * values[node] + 0.5 * values[node + 1]
            continuation = (coupon + expected) / (1.0 + rate * inputs["dt"])
            if callable_bond and step + 1 >= call_start:
                continuation = min(continuation, call_price)
            new_values.append(continuation)
        values = new_values
    return float(values[0])


def effective_duration(shock: float=0.0001) -> pd.DataFrame:
    """Approximate effective duration for callable and straight bonds."""
    inputs = load_tree_inputs()
    base_r0 = inputs["r0"]
    rows = []  # duration diagnostics
    for callable_bond in [False, True]:
        base = bond_value(callable_bond)
        up_value = bond_value(callable_bond, base_r0 + shock)
        down_value = bond_value(callable_bond, base_r0 - shock)
        duration = (down_value - up_value) / (2.0 * base * shock)
        rows.append({
            "bond": "callable" if callable_bond else "straight",
            "price": base,
            "duration": duration,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact chapter result summary."""
    print(effective_duration().round(4))


if __name__ == "__main__":
    main()
