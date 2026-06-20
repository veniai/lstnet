"""Tests for lstnet value objects (Site, GroundLST)."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from lstnet.models import GroundLST, Site


def test_site_construction_and_fields():
    s = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
    assert s.name == "psu"
    assert s.network == "SURFRAD"
    assert s.lon == pytest.approx(-77.93)
    assert s.lat == pytest.approx(40.72)


def test_site_is_frozen():
    s = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "dra"


def test_site_is_hashable():
    s = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
    assert hash(s) == hash(Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72))


def test_ground_lst_construction():
    site = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
    t = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)
    g = GroundLST(
        overpass_time=t,
        site=site,
        lst_k=305.32,
        emissivity=0.95,
        day_or_night="Unknown",
        qc_flag="OK",
    )
    assert g.lst_k == pytest.approx(305.32)
    assert g.emissivity == pytest.approx(0.95)
    assert g.qc_flag == "OK"
    assert g.day_or_night == "Unknown"
    assert g.overpass_time == t
    assert g.site is site
