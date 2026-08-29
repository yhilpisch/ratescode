"""
Python & AI for Rates, Bonds, and Credit
Appendix H · Mathematical Notation Reference

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Notation-domain map for Appendix H.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


NAVY = "#001F5B"
BLUE = "#2F6DB5"
LIGHT_BLUE = "#8CB6E8"
ORANGE = "#D9822B"
ROOT = Path(__file__).resolve().parents[2]


def draw_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    symbols: str,
    color: str,
) -> None:
    """Draw a rounded notation box with a title and math symbols."""
    width, height = size
    x_pos = center[0] - width / 2.0
    y_pos = center[1] - height / 2.0
    patch = FancyBboxPatch(
        (x_pos, y_pos),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=NAVY,
        facecolor=color,
        alpha=0.16,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1] + 0.028,
        title,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=10.2,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        center[0],
        center[1] - 0.030,
        symbols,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=11.2,
        transform=ax.transAxes,
    )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Draw a directional link between notation domains."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": NAVY,
            "lw": 1.8,
            "mutation_scale": 11,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def main() -> None:
    """Create a notation-domain overview figure."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    draw_box(
        ax,
        center=(0.17, 0.73),
        size=(0.20, 0.14),
        title="Cash flows",
        symbols=r"$c_i,\ D(0,T),\ P$",
        color=LIGHT_BLUE,
    )
    draw_box(
        ax,
        center=(0.50, 0.73),
        size=(0.22, 0.14),
        title="Rates models",
        symbols=r"$r_t,\ W_t,\ \mathbb{Q}$",
        color=BLUE,
    )
    draw_box(
        ax,
        center=(0.83, 0.73),
        size=(0.22, 0.14),
        title="Credit and risk",
        symbols=r"$s,\ \mathrm{CS01},\ L_t$",
        color=ORANGE,
    )
    draw_box(
        ax,
        center=(0.30, 0.28),
        size=(0.16, 0.14),
        title="ML",
        symbols=r"$X,\ y,\ \hat{y}$",
        color=LIGHT_BLUE,
    )
    draw_box(
        ax,
        center=(0.70, 0.28),
        size=(0.24, 0.14),
        title="GenAI workflows",
        symbols=r"$d,\ s(d),\ c_i,\ R$",
        color=ORANGE,
    )

    draw_arrow(ax, start=(0.27, 0.73), end=(0.39, 0.73))
    draw_arrow(ax, start=(0.61, 0.73), end=(0.72, 0.73))
    draw_arrow(ax, start=(0.46, 0.64), end=(0.35, 0.37))
    draw_arrow(ax, start=(0.83, 0.64), end=(0.70, 0.37))

    fig.tight_layout()
    fig.savefig(out / "appx_h_notation_domains.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
