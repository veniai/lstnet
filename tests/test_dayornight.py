"""Tests for day/night determination (astral-based port of methods/DayorNight.py)."""
from __future__ import annotations

from datetime import datetime, timezone

from lstnet.dayornight import dayornight
from lstnet.models import Site
from lstnet.sites import get_site


def test_day_summer_afternoon_utc_at_us_site():
    # psu (-77.93, 40.72); 2022-07-15 16:00 UTC ~= noon EDT -> Day
    psu = get_site("psu")
    t = datetime(2022, 7, 15, 16, 0, tzinfo=timezone.utc)
    assert dayornight(psu, t) == "Day"


def test_night_early_morning_utc_at_us_site():
    # psu; 2022-07-15 06:00 UTC ~= 2am EDT -> Night
    psu = get_site("psu")
    t = datetime(2022, 7, 15, 6, 0, tzinfo=timezone.utc)
    assert dayornight(psu, t) == "Night"


def test_polar_night_returns_night():
    # 80N near winter solstice -> sun always below horizon -> Night (no exception)
    polar = Site(name="north", network="SURFRAD", lon=0.0, lat=80.0)
    t = datetime(2022, 12, 21, 12, 0, tzinfo=timezone.utc)
    assert dayornight(polar, t) == "Night"


def test_returns_only_day_or_night():
    psu = get_site("psu")
    for hour in range(0, 24, 3):
        t = datetime(2022, 7, 15, hour, 0, tzinfo=timezone.utc)
        assert dayornight(psu, t) in {"Day", "Night"}
