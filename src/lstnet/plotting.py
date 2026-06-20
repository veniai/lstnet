"""Validation plots (matplotlib). Headless-safe: callers can save figures.

``scatter_plot`` draws retrieved-vs-ground with a 1:1 reference line;
``time_series_plot`` draws ground and retrieved LST over the overpass times.
"""
from __future__ import annotations

from datetime import datetime

import matplotlib

# Headless-safe: this module is for SAVING figures, not interactive display.
# Force Agg so importing it never triggers a GUI backend (e.g. Qt when
# PySide6 is installed) that needs a display. The GUI builds its own Qt canvas
# via FigureCanvasQTAgg and does not import this module.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lstnet.models import ValidationResult


def scatter_plot(result: ValidationResult, *, title: str = "Retrieved vs Ground LST"):
    """Return a matplotlib Figure of retrieved-vs-ground LST with a 1:1 line."""
    fig, ax = plt.subplots()
    if result.pairs:
        x = [p.ground.lst_k for p in result.pairs]
        y = [p.retrieved.lst_k for p in result.pairs]
        ax.scatter(x, y, zorder=3)
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1", zorder=2)
        ax.legend()
    ax.set_xlabel("Ground LST (K)")
    ax.set_ylabel("Retrieved LST (K)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def time_series_plot(result: ValidationResult, *, title: str = "LST time series"):
    """Return a matplotlib Figure of ground and retrieved LST over time."""
    fig, ax = plt.subplots()
    if result.pairs:
        times: list[datetime] = [p.ground.overpass_time for p in result.pairs]
        ax.plot(times, [p.ground.lst_k for p in result.pairs], "o-", label="ground")
        ax.plot(times, [p.retrieved.lst_k for p in result.pairs], "s-", label="retrieved")
        ax.legend()
    ax.set_xlabel("Overpass time (UTC)")
    ax.set_ylabel("LST (K)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
