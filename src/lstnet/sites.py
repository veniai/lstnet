"""Site registry: the 25 SURFRAD / PKULSTNet / HiWATER validation sites.

SURFRAD (7) is the online-accessible default network (NOAA serves its data);
PKULSTNet (7) and HiWATER (11) readers need local datasets the caller supplies.
Coordinates (lon, lat) ported verbatim from the legacy ``methods/site_map.py``.
No map drawing — that concern does not belong in the core library.
"""
from __future__ import annotations

from lstnet.models import Site

_SURFRAD: dict[str, tuple[float, float]] = {
    "gwn": (-89.87, 34.25),
    "dra": (-116.01947, 36.62373),
    "bon": (-88.37, 40.05),
    "tbl": (-105.24, 40.12),
    "psu": (-77.93, 40.72),
    "sxf": (-96.62, 43.73),
    "fpk": (-105.10, 48.31),
}

_PKULSTNET: dict[str, tuple[float, float]] = {
    "hnw": (110.0152, 19.68053333),
    "cqb": (106.3192833, 29.762728333),
    "xzl": (94.739460, 29.762580),
    "hnh": (114.31604333, 35.71568833),
    "imb": (111.209193, 41.353298333),
    "hbc": (117.24719, 42.41165),
    "xjf": (87.919188, 44.377822),
}

_HIWATER: dict[str, tuple[float, float]] = {
    "arz": (100.4643, 38.0473),
    "dmz": (100.3722, 38.8555),
    "dsl": (98.9406, 38.8399),
    "hhz": (100.4756, 38.8270),
    "hzz": (100.3201, 38.7659),
    "hmz": (100.9872, 42.1135),
    "hjl": (101.1335, 41.9903),
    "jyl": (101.1160, 37.8384),
    "sdq": (101.1374, 42.0012),
    "ykz": (100.2421, 38.0142),
    "zyz": (100.4464, 38.9751),
}


def _build(network: str, coords: dict[str, tuple[float, float]]) -> dict[str, Site]:
    return {
        name: Site(name=name, network=network, lon=lon, lat=lat)
        for name, (lon, lat) in coords.items()
    }


SITES: dict[str, Site] = {
    **_build("SURFRAD", _SURFRAD),
    **_build("PKULSTNet", _PKULSTNET),
    **_build("HiWATER", _HIWATER),
}


def get_site(name: str) -> Site:
    """Return the :class:`Site` for ``name``; raise ``KeyError`` if unknown."""
    try:
        return SITES[name]
    except KeyError:
        raise KeyError(f"Unknown site: {name!r}") from None


def by_network(network: str) -> list[Site]:
    """Return all sites belonging to ``network``."""
    return [site for site in SITES.values() if site.network == network]
