"""ASTER GED (Global Emissivity Database) emissivity source (Plan 1b open #2).

Provides ``AsterGEDEmissivity``, the spec's intended climatological-default
emissivity source (more stable than daily MODIS — it is a 2000–2008 mean of all
cloud-free ASTER scenes).

Retrieval path
--------------
The original Plan 1b brief named the **AppEEARS point API** as the access path.
That is **not available for ASTER GED**: as of 2026-06, AppEEARS' product
catalogue (176 products) does NOT include ``AG100`` / ``AG1km`` (ASTER GED) —
``GET /product`` returns no ASTER-GED entry and ``GET /product/AG100`` returns
``404 Product not found``. AppEEARS hosts daily MODIS/VIIRS/ECOSTRESS LST&E
products only. This was verified directly against the live API during
implementation.

The working path is the same as ``ModisDailyEmissivity``: **CMR granule search
via ``earthaccess`` → direct HDF5 download → ``h5py`` read**. AG100 V003 is a
1°×1° tile, ~100 m, HDF5, served from ``data.lpdaac.earthdatacloud.nasa.gov``
(Earthdata-auth-protected). The tile covering a point is named
``AG100.v003.<lat>.<lon>.0001`` (lat 0..89, lon 0..179, signed). Emissivity is
stored as int16 in ``Emissivity/Mean`` (shape ``(5, 1000, 1000)``, 5 ASTER TIR
bands 10–14) with scale ``0.001``; the per-pixel lat/lon grid is in
``Geolocation/{Latitude,Longitude}``.

Broadband conversion
--------------------
ASTER 5 narrowband emissivities → broadband (8–13.5 µm). Two formulas
(selectable via ``bbe``):

- ``'cheng_liang'`` (default) — Cheng & Liang (2014), J. Geophys. Res. Atmos.
  119:614-634, doi:10.1002/2013JD020689 (**open-access primary source**; 424
  spectra; R²=0.983). All five bands weighted::

      ε_bb = 0.197 + 0.025·ε10 + 0.057·ε11 + 0.237·ε12 + 0.333·ε13 + 0.146·ε14

- ``'ogawa'`` — Ogawa & Schmugge (2004), Earth Interactions 8(7) 3-band Sahara
  form (bands 13/14 ~0). **Secondary-sourced**: the paper PDF is persistently
  inaccessible (HTTP 403); coefficients reproduced from GLASS/CAMEL chains::

      ε_bb = 0.127·ε10 + 0.711·ε11 + 0.162·ε12

The default is Cheng & Liang because it is first-hand-verifiable and weights
all bands; Ogawa is kept for compatibility with older BBE products.

GDAL-free constraint
--------------------
No ``osgeo`` / ``gdal`` / ``pymodis``: AG100 ships HDF5, read with ``h5py``
(which bundles its own libhdf5). No ``eval``.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from lstnet import config
from lstnet.models import Site

# AG100 V003 product/version pins.
_SHORT_NAME = "AG100"
_VERSION = "003"

# Emissivity scaling (verified on the downloaded psu tile: DN 970 → 0.970).
_EMIS_SCALE = 0.001

# Fill value for the int16 emissivity SDS (ASTER GED convention).
_EMIS_FILL = -1

# Cheng & Liang (2014) ASTER→broadband (8–13.5 µm) coefficients, bands 10–14.
# Fit over 424 lab/field spectra (ASTER/UCSB), R²=0.983, RMSE=0.005. All five
# bands carry non-zero weight (bands 13/14 carry independent quartz/carbonate
# spectral info). Open-access primary source — coefficients read directly from
# the paper's §2.2, so this is a first-hand-verifiable citation.
# Cheng, J. & Liang, S. (2014), J. Geophys. Res. Atmos. 119:614-634,
# doi:10.1002/2013JD020689.
_CHENG_LIANG_COEFFS = (0.025, 0.057, 0.237, 0.333, 0.146)
_CHENG_LIANG_OFFSET = 0.197

# Ogawa & Schmugge (2004) 3-band Sahara form, bands 10–14 — kept as an
# alternative. NOTE: the Earth Interactions 8:7 PDF is persistently inaccessible
# (HTTP 403); these coefficients are reproduced from secondary sources (GLASS /
# CAMEL / MOD21 algorithm chains), NOT verified in an open primary full text.
# Bands 13/14 weight ~0 is an artefact of the Sahara/no-intercept subset, not a
# physical null. Prefer the Cheng & Liang default for verifiability.
# Ogawa, K. & Schmugge, T. (2004), Earth Interactions 8(7),
# doi:10.1175/1087-3562(2004)008<0001:MSBEOT>2.0.CO;2.
_OGAWA_COEFFS = (0.127, 0.711, 0.162, 0.0, 0.0)


# --- Injectable seams (so tests run offline) --------------------------------

class GranuleSearcher(Protocol):
    """Search CMR for the AG100 tile covering ``site``; return its HTTPS URL.

    Default implementation uses ``earthaccess.search_data`` with a point query.
    """

    def __call__(self, short_name: str, version: str, lon: float, lat: float) -> Optional[str]: ...


class GranuleDownloader(Protocol):
    """Download a granule URL into ``dest_dir`` and return the local path.

    Default implementation uses ``earthaccess.download``.
    """

    def __call__(self, url: str, dest_dir: Path) -> Path: ...


class H5Reader(Protocol):
    """Read the 5 ASTER emissivity DNs at the pixel nearest ``(lon, lat)``.

    Returns the raw int16 ``(b10, b11, b12, b13, b14)`` DNs. Default
    implementation uses ``h5py`` and does nearest-pixel lookup on the
    ``Geolocation`` grid.
    """

    def __call__(self, path: Path, lon: float, lat: float) -> tuple[int, int, int, int, int]: ...


# --- Main entry point -------------------------------------------------------

def retrieve_aster_ged_bands(
    site: Site,
    data_dir: Path | str | None = None,
    *,
    searcher: GranuleSearcher | None = None,
    downloader: GranuleDownloader | None = None,
    reader: H5Reader | None = None,
) -> tuple[float, float, float, float, float]:
    """Retrieve the 5 ASTER GED emissivities (bands 10–14) at ``site``.

    AG100 is a multi-year mean (2000–2008), so unlike daily MODIS it is
    time-independent: the ``overpass_time`` argument is not needed here and
    ``AsterGEDEmissivity.emissivity`` accepts it only to satisfy the
    :class:`EmissivitySource` protocol.

    Returns ``(nan,)*5`` (propagated to a ``nan`` BBE) when no tile is found or
    the pixel is a fill value — callers decide how to treat ``nan``.
    """
    nan5 = (float("nan"),) * 5
    # Force credential availability early when any default (online) seam is in
    # play; all-offline tests inject searcher+downloader and skip this.
    if searcher is None or downloader is None:
        config.earthdata_credentials()

    cache_dir = Path(data_dir) if data_dir is not None else (
        config.project_root() / "data" / "ASTER_GED"
    )

    hdf_path = _resolve_granule(site, cache_dir, searcher, downloader)
    if hdf_path is None:
        return nan5

    r = reader or _default_reader
    dn = r(hdf_path, site.lon, site.lat)
    if any(d == _EMIS_FILL for d in dn):
        return nan5
    return tuple(d * _EMIS_SCALE for d in dn)


def aster_to_broadband(
    em10: float, em11: float, em12: float, em13: float, em14: float, *, bbe: str = "cheng_liang"
) -> float:
    """ASTER 5 narrowband → broadband emissivity (8–13.5 µm).

    ``bbe``:
      - ``'cheng_liang'`` (default): Cheng & Liang (2014) 5-band, open-access
        primary source. All bands weighted.
      - ``'ogawa'``: Ogawa & Schmugge (2004) 3-band Sahara form (bands 13/14 ~0);
        secondary-sourced (the paper PDF is inaccessible).

    Returns ``nan`` if any input is ``nan`` (a missing/fill pixel).
    """
    if any(math.isnan(x) for x in (em10, em11, em12, em13, em14)):
        return float("nan")
    if bbe == "ogawa":
        c10, c11, c12, c13, c14 = _OGAWA_COEFFS
        return c10 * em10 + c11 * em11 + c12 * em12 + c13 * em13 + c14 * em14
    if bbe == "cheng_liang":
        c10, c11, c12, c13, c14 = _CHENG_LIANG_COEFFS
        return _CHENG_LIANG_OFFSET + c10 * em10 + c11 * em11 + c12 * em12 + c13 * em13 + c14 * em14
    raise ValueError(f"Unknown bbe formula {bbe!r}; use 'cheng_liang' or 'ogawa'.")


class AsterGEDEmissivity:
    """Broadband emissivity from the ASTER GED (AG100 V003) climatological mean.

    Wraps :func:`retrieve_aster_ged_bands` and converts the five ASTER TIR band
    emissivities to broadband via :func:`aster_to_broadband` (Ogawa 2004).

    Parameters
    ----------
    data_dir:
        Cache directory for downloaded tiles; defaults to
        ``project_root()/data/ASTER_GED``. Tiles are reused across calls.
    searcher / downloader / reader:
        Injectable seams (forwarded to :func:`retrieve_aster_ged_bands`) so the
        full chain (CMR search → HDF5 download → h5py read) is testable offline.
    """

    name = "aster_ged"

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        bbe: str = "cheng_liang",
        searcher: GranuleSearcher | None = None,
        downloader: GranuleDownloader | None = None,
        reader: H5Reader | None = None,
    ):
        if bbe not in ("cheng_liang", "ogawa"):
            raise ValueError(f"bbe must be 'cheng_liang' or 'ogawa', got {bbe!r}")
        self.data_dir = data_dir
        self.bbe = bbe
        self._searcher = searcher
        self._downloader = downloader
        self._reader = reader

    def emissivity(self, site: Site, overpass_time: datetime, day_or_night: str) -> float:
        # AG100 is a 2000–2008 mean — overpass_time / day_or_night are unused;
        # accepted only to satisfy the EmissivitySource protocol signature.
        bands = retrieve_aster_ged_bands(
            site, self.data_dir,
            searcher=self._searcher,
            downloader=self._downloader,
            reader=self._reader,
        )
        return aster_to_broadband(*bands, bbe=self.bbe)


# --- Default (online) seam implementations ----------------------------------

def _resolve_granule(
    site: Site,
    cache_dir: Path,
    searcher: GranuleSearcher | None,
    downloader: GranuleDownloader | None,
) -> Path | None:
    """Return the cached or freshly-downloaded AG100 tile path, or ``None``.

    Caches under ``cache_dir`` keyed by the granule filename (which encodes the
    1°×1° tile). Stale zero-byte partial downloads are treated as a cache miss.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Tile-predicted filename is an optimisation for cache hits only; the
    # authoritative tile comes from CMR. Check the predicted name first.
    predicted = _predicted_tile_filename(site.lon, site.lat)
    for cand in cache_dir.glob(f"{predicted}*.h5"):
        if cand.exists() and cand.stat().st_size > 0:
            return cand

    s = searcher or _default_searcher
    d = downloader or _default_downloader
    url = s(_SHORT_NAME, _VERSION, site.lon, site.lat)
    if url is None:
        return None
    return d(url, cache_dir)


def _predicted_tile_filename(lon: float, lat: float) -> str:
    """Predict the AG100 tile-name stem (cache-hit hint only).

    A 1°×1° tile covers ``[lon_floor, lon_floor+1) × [lat_floor, lat_floor+1)``
    and is named by its UPPER-LEFT (north-west) corner as seen in the granule
    id ``AG100.v003.<lat>.<lon>`` — verified on the psu tile ``41.-078``
    (covers lon [-78,-77), lat [40,41)):

      - ``lat``  = ``ceil(lat)``  (the tile's north edge)
      - ``lon``  = ``floor(lon)`` (the tile's west edge), formatted with sign
        and zero-padded to 3 digits (``-078``); lat is not padded.

    Authoritative tile still comes from CMR; this is only a cache-hit hint.
    """
    t_lat = int(math.ceil(lat))
    t_lon = int(math.floor(lon))
    # Longitude: sign + 3 zero-padded digits (e.g. -78 → "-078", 100 → "+100"?).
    # The downloaded psu tile is "41.-078"; eastern-hemisphere sign convention
    # is not directly exercised here but CMR is authoritative for cache misses.
    if t_lon < 0:
        lon_str = f"-{abs(t_lon):03d}"
    else:
        lon_str = f"{t_lon:03d}"
    return f"{_SHORT_NAME}.v{_VERSION}.{t_lat}.{lon_str}"


def _default_searcher(
    short_name: str, version: str, lon: float, lat: float
) -> Optional[str]:
    """CMR point search via ``earthaccess`` for the AG100 tile covering (lon,lat)."""
    import earthaccess  # local import: keeps the module importable offline

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=short_name, version=version,
        point=(lon, lat), count=1,
    )
    if not results:
        return None
    g = results[0]
    links = g.data_links(access="external") or g.data_links(access="on_prem")
    return links[0] if links else None


def _default_downloader(url: str, dest_dir: Path) -> Path:
    import earthaccess  # local import

    dest_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.download([url], local_path=str(dest_dir))
    name = url.rsplit("/", 1)[-1]
    path = dest_dir / name
    if not path.exists():
        # Fall back to the only .h5 in the dir if earthaccess renamed it.
        h5s = sorted(dest_dir.glob("*.h5"))
        if not h5s:
            raise FileNotFoundError(f"earthaccess did not produce {name} in {dest_dir}")
        path = h5s[0]
    return path


def _default_reader(path: Path, lon: float, lat: float) -> tuple[int, int, int, int, int]:
    """h5py read of the 5 emissivity DNs at the pixel nearest ``(lon, lat)``.

    Uses the per-pixel ``Geolocation/{Latitude,Longitude}`` grid (AG100 ships a
    full 1000×1000 lat/lon array, not a regular affine), so no projection math
    is needed.
    """
    import h5py  # local import
    import numpy as np  # local import

    with h5py.File(str(path), "r") as f:
        glat = f["Geolocation/Latitude"][:]
        glon = f["Geolocation/Longitude"][:]
        dist = (glon - lon) ** 2 + (glat - lat) ** 2
        row, col = np.unravel_index(int(np.argmin(dist)), dist.shape)
        em = f["Emissivity/Mean"][:, row, col]
    return tuple(int(v) for v in em)
