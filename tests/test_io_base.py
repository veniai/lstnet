"""Tests for the data-access interfaces (RadiationSample/Window, NetworkReader)."""
from __future__ import annotations

from datetime import datetime, timezone

from lstnet.io.base import NetworkReader, RadiationSample, RadiationWindow
from lstnet.models import Site

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
_TIME = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)


def test_radiation_types_construct():
    sample = RadiationSample(time=_TIME, l_up=400.0, l_down=300.0)
    window = RadiationWindow(site=_SITE, overpass_time=_TIME, samples=[sample], status="OK")
    assert window.status == "OK"
    assert window.samples[0].l_up == 400.0
    assert window.samples[0].l_down == 300.0


def test_fake_reader_satisfies_network_reader_protocol():
    class FakeReader:
        network = "FAKE"

        def read_radiation(self, site, overpass_time, window_minutes):
            return RadiationWindow(site, overpass_time, [], "OK")

    assert isinstance(FakeReader(), NetworkReader)
