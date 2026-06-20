"""MODIS MYD21A1D / MYD21A1N (C6.1) daily emissivity retrieval (Plan 1b).

GDAL-free retrieval of the MODIS bands-29/31/32 emissivity at a site on a
given overpass day. The verified stack (see ``.git/sdd/spike-1b-report.md``):

  - ``earthaccess``  for NASA Earthdata auth + CMR search + granule download.
  - ``pyhdf``        for HDF4 sub-dataset reads (ships its own libhdf4; no
                     system GDAL / ``osgeo`` / ``pymodis``).
  - manual sinusoidal pixel lookup (no rasterio — its manylinux wheel lacks
                     the HDF4 driver).

Two defects of the legacy ``methods/Modis_emiss.py`` are fixed here:

  - **QC ignored**  — the legacy code never read the ``QC`` SDS. This module
    reads it and reports ``qc='Fill'`` when the pixel DN is 0 (padded / outside
    valid swath) and ``qc='Missing'`` when QC flags the pixel as not produced
    (cloud / other) or the band data is out of valid range.
  - **Tile-edge padding crash** — psu sits on the eastern edge of h12v04; the
    legacy integer-rounding lookup could index into padding. Pixel indices are
    validated against ``[0, 1200)`` and padding is never returned as emissivity.

Network and pyhdf calls are exposed as injectable seams (``searcher`` /
``downloader`` / ``reader``) so the test-suite runs fully offline with crafted
fixtures.
"""
from __future__ import annotations

import dataclasses
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

from lstnet import config
from lstnet.models import Site

# --- Sinusoidal projection constants (MODIS standard) -----------------------
_R = 6371007.181  # MODIS sphere radius, metres
_TILE_WIDTH_M = (10.0 / 360.0) * 2.0 * math.pi * _R  # = 1111950.519667 m
_TILE_PIXELS = 1200  # MYD21A1D 1-km LST/emissivity grid per tile side
_PIXEL_SIZE_M = _TILE_WIDTH_M / _TILE_PIXELS

# MODIS MYD21A1D/C6.1 emissivity scaling (verified on the AppEEARS layer spec
# and on the spike HDF: DN * 0.002 + 0.49).
_EMIS_SCALE = 0.002
_EMIS_OFFSET = 0.49
_EMIS_FILL_DN = 0  # padded pixels outside the swath

_QC_MASK_MANDATORY = 0b11  # bits [0-1]: MODLAND_QA
_QC_GOOD = 0b00  # Pixel produced, good quality
_QC_OTHER_PROBLEM = 0b11  # Pixel not produced due to reasons other than cloud
_QC_CLOUD = 0b10  # Pixel not produced due to cloud
_QC_DATAQUAL_MASK = 0b1100  # bits [2-3]: Data Quality Flag
_QC_DATAQUAL_MISSING = 0b0100  # 01 (<<2) = Missing Pixel

# Version pin (C6.1, NOT the legacy C6 ``.006``).
_VERSION = "061"

# Minimum size for a cached granule to be trusted (a real MODIS HDF is 100 KB+).
# Smaller files (zero-byte partials, or placeholder bytes a test fake-downloader
# may leak into the cache dir) are treated as a cache miss and re-downloaded.
_MIN_CACHED_BYTES = 10_000


@dataclasses.dataclass(frozen=True)
class EmissivityBands:
    """Emissivity at the three MODIS TIR bands for one pixel + its QC verdict.

    ``qc`` is one of ``'OK'`` | ``'Fill'`` | ``'Missing'``:

      - ``'OK'``      — pixel produced (any quality band; values scaled).
      - ``'Fill'``    — pixel DN is the fill value (padding outside the swath);
                        emissivity fields are ``nan``.
      - ``'Missing'`` — QC marks the pixel as not produced (cloud / other
                        reason / missing-band) or a band DN is outside its valid
                        [1, 255] range; emissivity fields are ``nan``.
    """

    em29: float
    em31: float
    em32: float
    qc: str


# --- Injectable seams (so tests run offline) --------------------------------

class GranuleSearcher(Protocol):
    """Search CMR for the granule covering ``tile`` on ``date``.

    Returns the granule's HTTPS URL (data link), or ``None`` if no granule was
    found. Default implementation uses ``earthaccess.search_data``.
    """

    def __call__(
        self, short_name: str, version: str, tile: str, date: datetime
    ) -> Optional[str]: ...


class GranuleDownloader(Protocol):
    """Download a granule URL into ``dest_dir`` and return the local path.

    Default implementation uses ``earthaccess.download``. The result must exist
    on disk when this returns.
    """

    def __call__(self, url: str, dest_dir: Path) -> Path: ...


class HdfReader(Protocol):
    """Open an HDF at ``path`` and return the three band DNs + QC at ``(row,col)``.

    Returns ``(dn29, dn31, dn32, qc_word)`` as plain ints (qc_word may be 0 if
    the file has no QC SDS). Default implementation uses ``pyhdf``.
    """

    def __call__(
        self, path: Path, row: int, col: int
    ) -> tuple[int, int, int, int]: ...


# --- Tile geometry ----------------------------------------------------------

def modis_tile(site: Site) -> tuple[int, int]:
    """Return the ``(h, v)`` MODIS sinusoidal-tile indices for ``site``.

    Standard MODIS tiling: 36 tiles wide (h: 0..35), 18 tall (v: 0..17). Tile
    width = (10/360) * 2*pi * R. Forward transform follows the MODLAND tile
    calculator; verified against psu → h12v04 in the spike.
    """
    # Clamp latitude to the valid sinusoidal range [-90, 90].
    lat = max(-90.0, min(90.0, site.lat))
    sy = _R * math.radians(lat)
    sx = _R * math.radians(site.lon) * math.cos(math.radians(lat))
    # UL corner of the global grid = (-180 deg lon, +90 deg lat) mapped to the
    # sinusoidal plane: x = 0 (lon -180 -> x = -2*pi*R/2 = -pi*R, but MODIS tile
    # grid origin x0 = -_TILE_WIDTH_M * 18).
    x0 = -_TILE_WIDTH_M * 18.0
    y0 = _TILE_WIDTH_M * 9.0
    h = int((sx - x0) // _TILE_WIDTH_M)
    v = int((y0 - sy) // _TILE_WIDTH_M)
    h = max(0, min(35, h))
    v = max(0, min(17, v))
    return h, v


def _tile_string(h: int, v: int) -> str:
    return f"h{h:02d}v{v:02d}"


def _pixel_index(
    site: Site, ul_x: float, ul_y: float
) -> tuple[int, int]:
    """Sinusoidal forward transform → ``(row, col)`` inside the tile."""
    lat = max(-90.0, min(90.0, site.lat))
    sx = _R * math.radians(site.lon) * math.cos(math.radians(lat))
    sy = _R * math.radians(lat)
    col = int((sx - ul_x) / _PIXEL_SIZE_M)
    row = int((ul_y - sy) / _PIXEL_SIZE_M)
    return row, col


# --- Default (online) seam implementations ----------------------------------

def _default_searcher(
    short_name: str, version: str, tile: str, date: datetime
) -> Optional[str]:
    """CMR search via ``earthaccess`` for the granule covering ``tile`` on ``date``.

    Searches a +/-1-day window so a missing granule on the exact overpass day
    (common near the start/end of a 16-day composite or due to cloud) does not
    silently fail. Returns the first granule's HTTPS data link.
    """
    import earthaccess  # local import: keeps the module importable offline

    earthaccess.login(strategy="environment")
    h, v = int(tile[1:3]), int(tile[4:6])
    ul_lon, ul_lat = _tile_upper_left_lonlat(h, v)
    lr_lon, lr_lat = _tile_lower_right_lonlat(h, v)
    bbox = (min(ul_lon, lr_lon), min(ul_lat, lr_lat),
            max(ul_lon, lr_lon), max(ul_lat, lr_lat))
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    # +/-1 day window so a missing granule on the exact day does not silently
    # fail, but the requested AYYYYJJJ is preferred when present.
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        bounding_box=bbox,
        temporal=(start - timedelta(days=1), start + timedelta(days=2)),
    )
    if not results:
        return None
    wanted_date = f".A{start.strftime('%Y%j')}."
    candidates = [g for g in results if hasattr(g, "data_links")]
    https_links = lambda g: g.data_links(access="external") or g.data_links(access="on_prem")
    # Prefer: exact-date granule for the requested tile; else any granule for
    # the tile; else the first granule.
    chosen = None
    for g in candidates:
        for link in https_links(g):
            if wanted_date in link and tile in link:
                chosen = link
                break
        if chosen:
            break
    if chosen is None:
        for g in candidates:
            for link in https_links(g):
                if tile in link:
                    chosen = link
                    break
            if chosen:
                break
    if chosen is None and candidates:
        chosen = https_links(candidates[0])[0]
    return chosen


def _tile_upper_left_lonlat(h: int, v: int) -> tuple[float, float]:
    """Approximate (lon, lat) of the tile upper-left corner for CMR bbox search.

    Inverse of :func:`modis_tile`; accurate enough for the +/-tile-wide bbox
    that earthaccess uses to filter granules.
    """
    x0 = -_TILE_WIDTH_M * 18.0
    y0 = _TILE_WIDTH_M * 9.0
    ul_x = x0 + h * _TILE_WIDTH_M
    ul_y = y0 - v * _TILE_WIDTH_M
    lat = math.degrees(ul_y / _R)
    lon = math.degrees(ul_x / (_R * math.cos(math.radians(lat))))
    return lon, lat


def _tile_lower_right_lonlat(h: int, v: int) -> tuple[float, float]:
    ul_lon, ul_lat = _tile_upper_left_lonlat(h, v)
    lr_lon, lr_lat = _tile_upper_left_lonlat(h + 1, v + 1)
    return lr_lon, lr_lat


def _default_downloader(url: str, dest_dir: Path) -> Path:
    import earthaccess  # local import

    dest_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.download([url], local_path=str(dest_dir))
    name = url.rsplit("/", 1)[-1]
    path = dest_dir / name
    if not path.exists():
        # earthaccess sometimes appends a suffix; fall back to the only .hdf.
        hdhs = sorted(dest_dir.glob("*.hdf"))
        if not hdhs:
            raise FileNotFoundError(f"earthaccess did not produce {name} in {dest_dir}")
        path = hdhs[0]
    return path


def _default_reader(path: Path, row: int, col: int) -> tuple[int, int, int, int]:
    """pyhdf read of Emis_29/31/32 + QC at ``(row, col)``."""
    from pyhdf.SD import SD, SDC  # local import

    f = SD(str(path), SDC.READ)
    try:
        dn29 = int(f.select("Emis_29").get()[row, col])
        dn31 = int(f.select("Emis_31").get()[row, col])
        dn32 = int(f.select("Emis_32").get()[row, col])
        try:
            qc = int(f.select("QC").get()[row, col])
        except Exception:
            qc = 0
    finally:
        f.end()
    return dn29, dn31, dn32, qc


# --- Main entry point -------------------------------------------------------

def retrieve_emissivity_bands(
    site: Site,
    overpass_time: datetime,
    day_or_night: str,
    data_dir: Path | str | None = None,
    *,
    searcher: GranuleSearcher | None = None,
    downloader: GranuleDownloader | None = None,
    reader: HdfReader | None = None,
    strict: bool = False,
) -> EmissivityBands:
    """Retrieve Emis_29/31/32 + QC at ``site`` for the overpass day.

    Parameters mirror the SURFRAD reader's offline-testable design: pass a
    ``data_dir`` to control the cache, and override ``searcher`` / ``downloader``
    / ``reader`` in tests to avoid the network and pyhdf entirely. ``day_or_night``
    picks ``MYD21A1D`` (day) or ``MYD21A1N`` (night).
    """
    # When any default (online) seam is in play, force credential availability
    # early so a missing-env error surfaces before the first network call.
    # All-offline tests inject both searcher+downloader and skip this.
    if searcher is None or downloader is None:
        config.earthdata_credentials()

    if overpass_time.tzinfo is None:
        overpass_time = overpass_time.replace(tzinfo=timezone.utc)

    short_name = _short_name_for(day_or_night)
    h, v = modis_tile(site)
    tile = _tile_string(h, v)

    hdf_path = _resolve_granule(
        short_name, tile, overpass_time, data_dir, searcher, downloader
    )
    if hdf_path is None:
        return EmissivityBands(float("nan"), float("nan"), float("nan"), "Missing")

    row, col = _resolve_pixel(site, h, v, hdf_path, reader)
    if not (0 <= row < _TILE_PIXELS and 0 <= col < _TILE_PIXELS):
        return EmissivityBands(float("nan"), float("nan"), float("nan"), "Fill")

    r = reader or _default_reader
    dn29, dn31, dn32, qc_word = r(hdf_path, row, col)
    return _scale_and_qc(dn29, dn31, dn32, qc_word, strict=strict)


def _resolve_granule(
    short_name: str,
    tile: str,
    overpass_time: datetime,
    data_dir: Path | str | None,
    searcher: GranuleSearcher | None,
    downloader: GranuleDownloader | None,
) -> Path | None:
    """Return the cached or freshly-downloaded granule path, or ``None``.

    Caches under ``data_dir`` (or ``project_root()/data/MODIS``) keyed by
    ``{short_name}.A{YYYYJJJ}.{tile}.061.*.hdf``. Files that are missing,
    zero-byte, or too small to be a real MODIS granule (< ``_MIN_CACHED_BYTES``)
    are treated as a cache miss and re-fetched — this also discards bogus
    placeholder files a test fake-downloader may have written into the cache dir.
    """
    cache_dir = Path(data_dir) if data_dir is not None else (
        config.project_root() / "data" / "MODIS"
    )
    date_str = overpass_time.strftime("%Y%j")
    pattern = f"{short_name}.A{date_str}.{tile}.{_VERSION}.*.hdf"
    for cand in cache_dir.glob(pattern):
        if cand.exists() and cand.stat().st_size >= _MIN_CACHED_BYTES:
            return cand

    s = searcher or _default_searcher
    d = downloader or _default_downloader
    url = s(short_name, _VERSION, tile, overpass_time)
    if url is None:
        return None
    return d(url, cache_dir)


def _resolve_pixel(
    site: Site, h: int, v: int, hdf_path: Path, reader: HdfReader | None
) -> tuple[int, int]:
    """Return ``(row, col)`` inside the tile for ``site``.

    The default path reads ``UpperLeftPointMtrs`` from the HDF's
    ``StructMetadata.0``; the geometric UL (:func:`_tile_upper_left_metres`) is
    an exact match to that metadata (verified on the spike — sub-metre
    agreement), so when a reader is injected (offline tests with no real HDF)
    we use the geometric UL and skip pyhdf metadata access entirely.
    """
    if reader is None:
        ul_x, ul_y = _read_upper_left(hdf_path)
    else:
        ul_x, ul_y = _tile_upper_left_metres(h, v)
    return _pixel_index(site, ul_x, ul_y)


def _scale_and_qc(dn29: int, dn31: int, dn32: int, qc_word: int, *, strict: bool = False) -> EmissivityBands:
    """Apply the DN→emissivity scale and the QC + padding verdict.

    - DN == 0 (fill value) on any band      → ``qc='Fill'``.
    - QC marks pixel not produced / missing → ``qc='Missing'``.
    - Any band DN outside [1, 255]          → ``qc='Missing'``.
    - Otherwise                             → ``qc='OK'`` + scaled values.

    ``strict`` (default False) also rejects MODLAND_QA=01 (unreliable) pixels.
    """
    if _is_fill(dn29) or _is_fill(dn31) or _is_fill(dn32):
        return EmissivityBands(float("nan"), float("nan"), float("nan"), "Fill")
    if _is_qc_missing(qc_word, strict=strict):
        return EmissivityBands(float("nan"), float("nan"), float("nan"), "Missing")
    if not (_is_valid_dn(dn29) and _is_valid_dn(dn31) and _is_valid_dn(dn32)):
        return EmissivityBands(float("nan"), float("nan"), float("nan"), "Missing")
    return EmissivityBands(
        em29=dn29 * _EMIS_SCALE + _EMIS_OFFSET,
        em31=dn31 * _EMIS_SCALE + _EMIS_OFFSET,
        em32=dn32 * _EMIS_SCALE + _EMIS_OFFSET,
        qc="OK",
    )


def _short_name_for(day_or_night: str) -> str:
    key = (day_or_night or "").strip().lower()
    if key == "day":
        return "MYD21A1D"
    if key == "night":
        return "MYD21A1N"
    raise ValueError(
        f"day_or_night must be 'Day' or 'Night', got {day_or_night!r}"
    )


def _tile_upper_left_metres(h: int, v: int) -> tuple[float, float]:
    x0 = -_TILE_WIDTH_M * 18.0
    y0 = _TILE_WIDTH_M * 9.0
    return x0 + h * _TILE_WIDTH_M, y0 - v * _TILE_WIDTH_M


def _read_upper_left(path: Path) -> tuple[float, float]:
    """Read ``UpperLeftPointMtrs`` from the HDF ``StructMetadata.0``.

    StructMetadata stores only the upper-left for tile-scoped MYD21 products
    (verified in the spike); the lower-right is derived from standard MODIS
    tile geometry but is not needed for the forward pixel lookup.
    """
    from pyhdf.SD import SD, SDC

    f = SD(str(path), SDC.READ)
    try:
        meta = f.attributes().get("StructMetadata.0", "")
    finally:
        f.end()
    m = re.search(
        r"UpperLeftPointMtrs=\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)", meta
    )
    if not m:
        raise ValueError(f"UpperLeftPointMtrs not found in {path}")
    return float(m.group(1)), float(m.group(2))


def _is_fill(dn: int) -> bool:
    return dn == _EMIS_FILL_DN


def _is_valid_dn(dn: int) -> bool:
    return 1 <= dn <= 255


def _is_qc_missing(qc_word: int, *, strict: bool = False) -> bool:
    """True when QC marks the pixel as not-produced (cloud/other) or missing.

    Bit decoding (MYD21 QC legend, right-to-left):
      [0-1] MODLAND_QA:  00 good, 01 unreliable (still produced — accept),
                        10 not produced (cloud), 11 not produced (other).
      [2-3] Data Quality: 01 = Missing Pixel.

    ``strict`` (default False) also treats MODLAND_QA=01 (unreliable) as missing.
    """
    mandatory = qc_word & _QC_MASK_MANDATORY
    if strict and mandatory != _QC_GOOD:
        return True
    if mandatory in (_QC_CLOUD, _QC_OTHER_PROBLEM):
        return True
    if (qc_word & _QC_DATAQUAL_MASK) == _QC_DATAQUAL_MISSING:
        return True
    return False
