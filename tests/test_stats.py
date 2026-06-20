"""Tests for validation statistics (bias / rmse / pearson_r / regression)."""
from __future__ import annotations

import math

from lstnet.models import GroundLST, RetrievedLST, Site, ValidationPair
from lstnet.stats import (
    bias,
    compute_stats,
    linear_regression,
    pearson_r,
    rmse,
)

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)


def _pair(ground_k: float, retr_k: float) -> ValidationPair:
    from datetime import datetime, timezone

    t = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)
    g = GroundLST(t, _SITE, ground_k, 0.97, "Day", "OK")
    r = RetrievedLST(t, _SITE, retr_k, "algo")
    return ValidationPair(ground=g, retrieved=r, diff=retr_k - ground_k)


def test_bias_mean_of_diffs():
    pairs = [_pair(300, 301), _pair(300, 299), _pair(300, 302), _pair(300, 298)]
    # diffs: +1, -1, +2, -2 -> mean 0
    assert bias(pairs) == pytest_approx(0.0)


def test_rmse_of_diffs():
    pairs = [_pair(300, 301), _pair(300, 299)]  # diffs +1, -1
    assert rmse(pairs) == pytest_approx(1.0)


def test_pearson_r_perfect_positive():
    # retrieved = ground + 5 exactly -> r == 1
    pairs = [_pair(290, 295), _pair(295, 300), _pair(300, 305), _pair(305, 310)]
    assert pearson_r(pairs) == pytest_approx(1.0)


def test_pearson_r_zero_for_no_correlation():
    # ground ascending, retrieved scrambled with no linear relation -> r ~= 0
    pairs = [_pair(290, 300), _pair(295, 290), _pair(300, 305), _pair(305, 295)]
    assert abs(pearson_r(pairs)) < 1e-9


def test_linear_regression_returns_slope_intercept():
    # retrieved = 2*ground + 10
    pairs = [_pair(290, 590), _pair(295, 600), _pair(300, 610)]
    slope, intercept = linear_regression(pairs)
    assert slope == pytest_approx(2.0)
    assert intercept == pytest_approx(10.0)


def test_compute_stats_aggregates_all():
    pairs = [_pair(290, 295), _pair(295, 300), _pair(300, 305)]  # perfect r=1, diff=+5
    stats = compute_stats(pairs)
    assert stats.n == 3
    assert stats.bias == pytest_approx(5.0)
    assert stats.rmse == pytest_approx(5.0)
    assert stats.r == pytest_approx(1.0)
    assert stats.slope == pytest_approx(1.0)  # retrieved vs ground slope (y=x+5)
    assert stats.intercept == pytest_approx(5.0)


def pytest_approx(value, rel=1e-9, abs=1e-9):
    import pytest

    return pytest.approx(value, rel=rel, abs=abs)
