"""Emissivity sources.

Three broadband-emissivity sources, by increasing fidelity:

- :class:`FixedEmissivity` — a constant (e.g. 0.95). Offline, zero deps. Use as a
  fallback only; carries surface-dependent bias (~1-2 K over vegetation/water).
- :class:`ModisDailyEmissivity` — MYD21A1D/A1N C6.1 *daily* emissivity (via
  :mod:`lstnet.io.modis`, earthaccess + pyhdf). Time-varying; needs Earthdata
  credentials.
- :class:`AsterGEDEmissivity` — ASTER GED (AG100 V003) *climatological* mean
  (via :mod:`lstnet.io.aster_ged`, CMR + h5py). Multi-year stable; recommended
  for sub-K validation (matches the CEOS LPV best practice of a stable
  reference emissivity).

There is **no implicit default** on :func:`lstnet.ground_lst.compute_ground_lst`
— ``emiss_src`` is required, so the library never silently hits the network.
Recommendation: ``AsterGEDEmissivity`` for validation accuracy,
``ModisDailyEmissivity`` when day-to-day emissivity variation matters,
``FixedEmissivity(0.95)`` for offline / no-credential runs.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from lstnet.io.base import EmissivitySource
from lstnet.io.modis import EmissivityBands, retrieve_emissivity_bands
from lstnet.models import Site


class FixedEmissivity:
    """Constant emissivity for all sites/times.

    Valid range [0.49, 1.0] mirrors the MODIS emissivity physical scale floor.
    Note: a fixed value carries surface-dependent bias (vegetation/water ~0.985,
    bare soil ~0.92-0.95); use it as a fallback, not for sub-K validation.
    """

    name = "fixed"

    def __init__(self, value: float):
        if not 0.49 <= value <= 1.0:
            raise ValueError(
                f"FixedEmissivity value must be in [0.49, 1.0], got {value}"
            )
        self.value = value

    def emissivity(self, site: Site, overpass_time: datetime, day_or_night: str) -> float:
        return self.value


# --- Broadband-emissivity formulas from MODIS bands 29/31/32 ----------------
#
# Two formulas are supported:
#
#  * ``'legacy'`` (DEPRECATED, bare-soil only):  ``0.329·em29 + 0.572·em31 + 0.095``
#    This is exactly the regression used by the legacy ``methods/Modis_emiss.py``
#    (``0.329·em29 + 0.572·em31 + 0.095`` — em32 was unused there). It is a
#    bare-soil-only formula and overestimates BBE over vegetation/water. Kept
#    solely for reproducing historical results; new code should use ``'modern'``.
#
#  * ``'modern'`` (default):  ``0.047·em29 + 0.977·em31 − 0.024·em32``
#    Broadband emissivity over the 8–13.5 µm thermal window from MODIS
#    narrowband emissivities (MYD21 TES, bands 29/31/32). Coefficients from
#    Ogawa, Schmugge, Jacob & French (2004), Remote Sensing of Environment
#    88(1-2):71-84, doi:10.1016/j.rse.2003.12.014 — the most-cited source and
#    the one endorsed by the MOD21 ATBD. Single global formula; per-class
#    regressions (water/vegetation/snow/bare-soil) differ by <0.005 BBE except
#    bare soil (~0.01, ~0.5–0.7 K LST) — add a bare-soil branch if validating
#    arid sites. Note ε32 has a NEGATIVE coefficient (suppresses the longer-
#    wavelength contribution); band 31 dominates.
_MODERN_COEFFS = (0.047, 0.977, -0.024)
_LEGACY_COEFFS = (0.329, 0.572, 0.0)
_LEGACY_OFFSET = 0.095


def broadband_emissivity(
    bands: EmissivityBands, bbe: str = "modern"
) -> float:
    """Compute broadband emissivity from :class:`EmissivityBands`.

    ``bbe`` selects the formula (see module docstring). Returns ``nan`` when
    ``bands.qc`` is not ``'OK'`` so callers can decide_qc on the result.
    """
    if bands.qc != "OK":
        return float("nan")
    if bbe == "legacy":
        # Legacy bare-soil regression: only em29/em31 carry weight.
        return _LEGACY_COEFFS[0] * bands.em29 + _LEGACY_COEFFS[1] * bands.em31 + _LEGACY_OFFSET
    if bbe == "modern":
        c29, c31, c32 = _MODERN_COEFFS
        return c29 * bands.em29 + c31 * bands.em31 + c32 * bands.em32
    raise ValueError(f"Unknown bbe formula {bbe!r}; use 'modern' or 'legacy'.")


class ModisDailyEmissivity:
    """Broadband emissivity from MYD21A1D/A1N C6.1 daily emissivity.

    Wraps :func:`lstnet.io.modis.retrieve_emissivity_bands` and converts the
    three band emissivities to broadband via :func:`broadband_emissivity`.

    Parameters
    ----------
    data_dir:
        Cache directory for downloaded granules; defaults to
        ``project_root()/data/MODIS``. Granules are reused across calls.
    bbe:
        ``'modern'`` (default, recommended) or ``'legacy'`` (bare-soil-only,
        deprecated; for reproducing old results).
    """

    name = "modis_daily"

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        bbe: str = "modern",
        searcher=None,
        downloader=None,
        reader=None,
        strict: bool = False,
    ):
        if bbe not in ("modern", "legacy"):
            raise ValueError(f"bbe must be 'modern' or 'legacy', got {bbe!r}")
        self.data_dir = data_dir
        self.bbe = bbe
        self.strict = strict
        # Injectable seams — forwarded to retrieve_emissivity_bands so the full
        # chain (CMR search → granule download → pyhdf read) is testable offline.
        self._searcher = searcher
        self._downloader = downloader
        self._reader = reader

    def emissivity(self, site: Site, overpass_time: datetime, day_or_night: str) -> float:
        bands = retrieve_emissivity_bands(
            site, overpass_time, day_or_night, self.data_dir,
            searcher=self._searcher,
            downloader=self._downloader,
            reader=self._reader,
            strict=self.strict,
        )
        return broadband_emissivity(bands, self.bbe)
