"""Day/night determination for a satellite overpass (port of methods/DayorNight.py).

Uses ``astral`` sun elevation: above horizon (elevation > 0) -> ``'Day'`` else
``'Night'``. This is robust to the UTC date-boundary quirk that astral's
sunrise/sunset events exhibit at western longitudes, and handles polar
night/day without special-casing.
"""
from __future__ import annotations

from datetime import datetime, timezone

from astral import LocationInfo
from astral.sun import elevation

from lstnet.models import Site


def dayornight(site: Site, overpass_time: datetime) -> str:
    """Return ``'Day'`` or ``'Night'`` for ``overpass_time`` (UTC) at ``site``."""
    if overpass_time.tzinfo is None:
        overpass_time = overpass_time.replace(tzinfo=timezone.utc)
    observer = LocationInfo("site", "region", "UTC", site.lat, site.lon).observer
    return "Day" if elevation(observer, overpass_time) > 0 else "Night"
