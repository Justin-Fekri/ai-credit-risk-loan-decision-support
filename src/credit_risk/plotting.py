"""Shared chart styling.

Keeping the palette and the axis treatment in one place means every figure the
project produces looks like it came from the same system -- which matters when
the charts end up in a README or a model review pack.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write straight to PNG; no GUI needed
import matplotlib.pyplot as plt  # noqa: E402

# A restrained palette: one accent for the model, grey for reference lines,
# and a warm tone reserved for anything the reader should treat as a warning.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
ACCENT = "#2b6cb0"
ACCENT_LIGHT = "#7da7d9"
WARN = "#b7791f"

RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
}


def style():
    """Use as a context manager so the settings never leak into a user's session."""
    return plt.rc_context(RC)


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
