"""Data-access interfaces (protocols) for lstnet.

These define the boundaries between the domain core and its data sources.
Concrete implementations live alongside (e.g. ``emissivity.py``, ``surfrad.py``).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Protocol, runtime_checkable

from lstnet.models import Site


@dataclasses.dataclass
class RadiationSample:
    """One radiation measurement: up/down longwave at an instant."""

    time: datetime
    l_up: float
    l_down: float


@dataclasses.dataclass
class RadiationWindow:
    """Radiation samples around an overpass moment, plus a reader status.

    ``status`` is one of: ``OK`` | ``NoData`` | ``FileNotFound``.
    """

    site: Site
    overpass_time: datetime
    samples: list[RadiationSample]
    status: str


@runtime_checkable
class NetworkReader(Protocol):
    """Reads station radiation data around a satellite overpass time."""

    network: str

    def read_radiation(
        self, site: Site, overpass_time: datetime, window_minutes: int
    ) -> RadiationWindow:
        """Return up/down longwave samples within +/- ``window_minutes`` of the overpass."""
        ...


@runtime_checkable
class EmissivitySource(Protocol):
    """A source of broadband surface emissivity for the ground-LST formula."""

    name: str

    def emissivity(self, site: Site, overpass_time: datetime, day_or_night: str) -> float:
        """Return broadband emissivity at ``site`` for the given overpass moment."""
        ...
