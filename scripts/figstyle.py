#!/usr/bin/env python3
"""Shared look-and-feel for the repo's README figures.

Every figure in assets/ imports this module, so the whole set reads as one
system: same palette, same type scale, same spine/grid treatment, same dpi.
Change a value here and every figure moves together.

Palette: two role colors (navy = client side, orange = Orin side) plus a
three-step severity ramp (green/amber/red) reused by the latency percentiles
and the success heatmap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- ink ----
INK = "#111827"
MUTED = "#6B7280"
FAINT = "#E5E7EB"

# ---- role colors ----
NAVY = "#1F3A5F"      # client / host side
ORANGE = "#C2410C"    # Jetson / shipped recipe
BLUE = "#1D4ED8"      # plain data series
GREEN = "#15803D"     # p50 / good
AMBER = "#B45309"     # p95 / caution
RED = "#B91C1C"       # p99 / loss

# ---- panel fills ----
PANEL_NAVY = "#F3F7FC"
PANEL_ORANGE = "#FDF5EE"
PANEL_GRAY = "#F9FAFB"

DPI = 180


def apply():
    """Install the shared rcParams. Call once at the top of a figure script."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 9,
        "axes.titlelocation": "left",
        "axes.grid": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.2,
    })


def tidy(ax, grid_axis=None):
    """Drop top/right spines and (optionally) add a faint grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=FAINT, lw=0.7, zorder=0)
        ax.set_axisbelow(True)


def panel(ax, x, y, w, h, fc, ec):
    """Rounded panel in data coordinates (used by the topology figure)."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.55,rounding_size=1.1",
        fc=fc, ec=ec, lw=1.0, zorder=1))


def text(ax, x, y, s, size=10, color=INK, ha="center", weight="normal", z=3):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            weight=weight, zorder=z)
