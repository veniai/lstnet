"""Tests for the lstnet site registry."""
from __future__ import annotations

import pytest

from lstnet.models import Site
from lstnet.sites import SITES, by_network, get_site


def test_total_count():
    assert len(SITES) == 25


def test_per_network_counts():
    assert len(by_network("SURFRAD")) == 7
    assert len(by_network("PKULSTNet")) == 7
    assert len(by_network("HiWATER")) == 11


def test_spot_check_coords_match_legacy():
    psu = get_site("psu")
    assert psu.network == "SURFRAD"
    assert psu.lon == pytest.approx(-77.93)
    assert psu.lat == pytest.approx(40.72)

    hbc = get_site("hbc")
    assert hbc.network == "PKULSTNet"
    assert hbc.lon == pytest.approx(117.24719)
    assert hbc.lat == pytest.approx(42.41165)

    arz = get_site("arz")
    assert arz.network == "HiWATER"
    assert arz.lon == pytest.approx(100.4643)
    assert arz.lat == pytest.approx(38.0473)


def test_get_site_returns_site_instance():
    s = get_site("gwn")
    assert isinstance(s, Site)
    assert s.name == "gwn"


def test_unknown_site_raises_keyerror():
    with pytest.raises(KeyError):
        get_site("does-not-exist")
