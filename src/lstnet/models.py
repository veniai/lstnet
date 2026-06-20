"""Value objects passed between lstnet modules."""
from __future__ import annotations

import dataclasses
from datetime import datetime


@dataclasses.dataclass(frozen=True)
class Site:
    """A ground validation site."""

    name: str
    network: str
    lon: float
    lat: float


@dataclasses.dataclass
class GroundLST:
    """Ground-truth LST at one satellite overpass moment."""

    overpass_time: datetime
    site: Site
    lst_k: float
    emissivity: float
    day_or_night: str
    qc_flag: str


@dataclasses.dataclass(frozen=True)
class RetrievedLST:
    """A retrieved-LST value (from the user's own algorithm or a satellite product)."""

    overpass_time: datetime
    site: Site
    lst_k: float
    source: str = "own-algorithm"


@dataclasses.dataclass(frozen=True)
class ValidationPair:
    """One matched ground-truth + retrieved-LST observation."""

    ground: GroundLST
    retrieved: RetrievedLST
    diff: float  # retrieved.lst_k - ground.lst_k


@dataclasses.dataclass(frozen=True)
class ValidationStats:
    """Aggregate validation statistics over a set of pairs."""

    n: int
    bias: float
    rmse: float
    r: float
    slope: float
    intercept: float


@dataclasses.dataclass
class ValidationResult:
    """Outcome of a validation run: matched pairs + unmatched + stats."""

    pairs: list
    unmatched_ground: list
    unmatched_retrieved: list
    stats: ValidationStats
