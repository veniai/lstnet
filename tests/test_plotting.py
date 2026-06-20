"""Tests for validation plots (headless Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from datetime import datetime, timezone  # noqa: E402

from lstnet.models import (  # noqa: E402
    GroundLST,
    RetrievedLST,
    Site,
    ValidationPair,
    ValidationResult,
    ValidationStats,
)
from lstnet.plotting import scatter_plot, time_series_plot  # noqa: E402

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)


def _result() -> ValidationResult:
    t = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)
    pairs = [
        ValidationPair(
            ground=GroundLST(t, _SITE, 300.0, 0.97, "Day", "OK"),
            retrieved=RetrievedLST(t, _SITE, 302.0, "algo"),
            diff=2.0,
        ),
        ValidationPair(
            ground=GroundLST(t, _SITE, 305.0, 0.97, "Day", "OK"),
            retrieved=RetrievedLST(t, _SITE, 304.0, "algo"),
            diff=-1.0,
        ),
    ]
    stats = ValidationStats(n=2, bias=0.5, rmse=1.5, r=0.9, slope=1.0, intercept=2.0)
    return ValidationResult(pairs=pairs, unmatched_ground=[], unmatched_retrieved=[], stats=stats)


def test_scatter_plot_returns_figure_with_1to1_line():
    fig = scatter_plot(_result())
    assert fig is not None
    ax = fig.axes[0]
    # one scatter collection + the 1:1 line
    assert len(ax.collections) >= 1
    labels = [line.get_label() for line in ax.get_lines()]
    assert "1:1" in labels


def test_time_series_plot_returns_figure():
    fig = time_series_plot(_result())
    assert fig is not None
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "ground" in labels and "retrieved" in labels


def test_plots_handle_empty_result():
    empty = ValidationResult(
        pairs=[], unmatched_ground=[], unmatched_retrieved=[],
        stats=ValidationStats(n=0, bias=0, rmse=0, r=0, slope=0, intercept=0),
    )
    assert scatter_plot(empty) is not None
    assert time_series_plot(empty) is not None
