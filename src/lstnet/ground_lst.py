"""Ground-truth LST physics.

Inverts the surface longwave radiance balance to recover surface temperature:
    L_up = emiss * sigma * T^4 + (1 - emiss) * L_down
    T    = ((L_up - (1 - emiss) * L_down) / (emiss * sigma)) ** 0.25
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from lstnet.dayornight import dayornight
from lstnet.io.base import EmissivitySource, NetworkReader
from lstnet.models import GroundLST, Site
from lstnet.qc import QC_NO_DATA, QC_OK, QC_STD_ERROR, QC_TIME_ERROR, decide_qc

# Stefan-Boltzmann constant (W m^-2 K^-4). Unified across networks — the legacy
# code used 5.67e-8 (SURFRAD) and 5.6697e-8 (HiWATER) inconsistently.
SIGMA = 5.670374e-8

_OVERPASS_FORMAT = "%Y%m%d%H%M"


def lst_from_radiance(
    l_up: float, l_down: float, emiss: float, sigma: float = SIGMA
) -> float:
    """Convert up/down longwave radiance + emissivity to LST in Kelvin.

    ``l_up``/``l_down`` are upwelling/downwelling longwave radiation (W/m^2).
    Raises :class:`ValueError` on non-physical input.
    """
    if not 0.0 < emiss <= 1.0:
        raise ValueError(f"emissivity must be in (0, 1], got {emiss}")
    reflected = (1.0 - emiss) * l_down
    net = l_up - reflected
    if net <= 0:
        raise ValueError(
            f"non-physical input: l_up ({l_up}) must exceed (1-emiss)*l_down ({reflected})"
        )
    return (net / (emiss * sigma)) ** 0.25


def parse_overpass_time(s: str) -> datetime:
    """Parse a 12-digit ``YYYYMMDDHHMM`` stamp into an aware UTC datetime.

    Raises :class:`ValueError` if ``s`` is not a 12-digit string or does not
    form a valid calendar datetime.
    """
    if not isinstance(s, str) or len(s) != 12 or not s.isdigit():
        raise ValueError(f"overpass_time must be 12 digits YYYYMMDDHHMM, got {s!r}")
    return datetime.strptime(s, _OVERPASS_FORMAT).replace(tzinfo=timezone.utc)


def compute_ground_lst(
    site: Site,
    overpass_time: datetime | str,
    emiss_src: EmissivitySource,
    reader: NetworkReader,
    window_minutes: int = 0,
) -> GroundLST:
    """Orchestrate reader + emissivity + physics + QC into one :class:`GroundLST`.

    - ``overpass_time`` as ``str`` is parsed via :func:`parse_overpass_time`;
      an unparseable stamp yields a ``TimeError`` result (no exception).
    - ``day_or_night`` is computed via :func:`lstnet.dayornight.dayornight`
      (astral sun elevation); ``"Unknown"`` only on a TimeError result.
    - Non-``OK`` reader windows propagate their status verbatim as the qc_flag.
    - Non-physical samples (``lst_from_radiance`` raises) are skipped; the
      remaining samples feed :func:`lstnet.qc.decide_qc`.

    ``window_minutes`` default ``0`` is a **sentinel**: each reader uses its
    native legacy window — SURFRAD applies ``range(row-5+step, row+6-step)``
    (±4 min for step=1 / 9 samples, ±2 min for step=3 / 5 samples), PKU uses
    its per-interval (1/2/3-min) branch, HiWATER uses the 2 nearest samples.
    A network-agnostic uniform window cannot reproduce all three legacy
    retrievals, so the default delegates. Pass an explicit positive value to
    override (SURFRAD honors it; PKU/HiWATER ignore it by design).
    """
    day_or_night = "Unknown"

    if isinstance(overpass_time, str):
        try:
            overpass_time = parse_overpass_time(overpass_time)
        except ValueError:
            return GroundLST(
                overpass_time=datetime(1, 1, 1, tzinfo=timezone.utc),
                site=site,
                lst_k=math.nan,
                emissivity=math.nan,
                day_or_night=day_or_night,
                qc_flag=QC_TIME_ERROR,
            )

    day_or_night = dayornight(site, overpass_time)
    emiss = emiss_src.emissivity(site, overpass_time, day_or_night)
    window = reader.read_radiation(site, overpass_time, window_minutes)

    if window.status != QC_OK:
        return GroundLST(
            overpass_time=overpass_time,
            site=site,
            lst_k=math.nan,
            emissivity=emiss,
            day_or_night=day_or_night,
            qc_flag=window.status,
        )

    lst_samples: list[float] = []
    for sample in window.samples:
        try:
            lst_samples.append(
                lst_from_radiance(sample.l_up, sample.l_down, emiss)
            )
        except ValueError:
            # Non-physical sample (e.g. l_up <= (1-emiss)*l_down) — skip.
            continue

    avg, flag = decide_qc(lst_samples)
    lst_k = avg if avg is not None else math.nan
    return GroundLST(
        overpass_time=overpass_time,
        site=site,
        lst_k=lst_k,
        emissivity=emiss,
        day_or_night=day_or_night,
        qc_flag=flag,
    )
