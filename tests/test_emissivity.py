"""Tests for the emissivity interface and FixedEmissivity."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lstnet.io.base import EmissivitySource
from lstnet.io.emissivity import FixedEmissivity
from lstnet.models import Site

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
_TIME = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)


def test_fixed_emissivity_returns_value():
    fe = FixedEmissivity(0.95)
    assert fe.name == "fixed"
    assert fe.emissivity(_SITE, _TIME, "Unknown") == pytest.approx(0.95)


def test_fixed_emissivity_boundary_values_accepted():
    # [0.49, 1.0] inclusive
    assert FixedEmissivity(0.49).emissivity(_SITE, _TIME, "Unknown") == pytest.approx(0.49)
    assert FixedEmissivity(1.0).emissivity(_SITE, _TIME, "Unknown") == pytest.approx(1.0)


def test_fixed_emissivity_invalid_raises():
    with pytest.raises(ValueError):
        FixedEmissivity(0.3)
    with pytest.raises(ValueError):
        FixedEmissivity(1.5)
    with pytest.raises(ValueError):
        FixedEmissivity(0.0)


def test_fixed_emissivity_satisfies_protocol():
    fe = FixedEmissivity(0.95)
    assert isinstance(fe, EmissivitySource)
