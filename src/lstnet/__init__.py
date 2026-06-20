"""lstnet: pure-Python land surface temperature library.

Ground-truth LST production (Plan 1a) + MODIS daily emissivity (Plan 1b) +
validation engine (Plan 1c). Public API for computing ground-truth LST at
satellite overpass times and validating it against retrieved LST.
"""
from __future__ import annotations

__version__ = "0.1.0"

from lstnet.dayornight import dayornight
from lstnet.ground_lst import SIGMA, compute_ground_lst, lst_from_radiance
from lstnet.io.aster_ged import AsterGEDEmissivity
from lstnet.io.emissivity import FixedEmissivity, ModisDailyEmissivity
from lstnet.models import (
    GroundLST,
    RetrievedLST,
    Site,
    ValidationPair,
    ValidationResult,
    ValidationStats,
)
from lstnet.sites import SITES, by_network, get_site
from lstnet.validation import TableRetrievedLST, validate

__all__ = [
    "SIGMA",
    "compute_ground_lst",
    "lst_from_radiance",
    "FixedEmissivity",
    "ModisDailyEmissivity",
    "AsterGEDEmissivity",
    "dayornight",
    "GroundLST",
    "Site",
    "SITES",
    "by_network",
    "get_site",
    "validate",
    "TableRetrievedLST",
    "RetrievedLST",
    "ValidationPair",
    "ValidationResult",
    "ValidationStats",
]
