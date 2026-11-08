"""
Python & AI for Rates, Bonds, and Credit
Part V · Credit Figure Layout

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Shared layouts for the Part V book figures and slide panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt

NAVY = "#001F5B"
BLUE = "#2F6DB5"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[1]


def legend_above(ax, columns: int=2) -> None:
    """Keep legends outside the plotting area."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=columns, frameon=False, fontsize=10)


def export_panels(stem: str, panels: list, book_panels: list | None=None):
    """Save the book composite and larger individual slide panels."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    composite = panels[:2] if book_panels is None else book_panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (_, title, draw) in zip(axes, composite):
        draw(ax)
        ax.set_title(title, color=NAVY, pad=30)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    for suffix, _, draw in panels:
        fig, ax = plt.subplots(figsize=(7.8, 2.7))
        draw(ax)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_{suffix}_slides.pdf")
        plt.close(fig)


def style_axis(ax) -> None:
    """Apply the established navy labels and remove unused spines."""
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.spines[["top", "right"]].set_visible(False)
