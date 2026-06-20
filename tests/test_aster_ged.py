"""Tests for the ASTER GED (AG100 V003) emissivity source (Plan 1b open #2).

Coverage:
  - Mocked (CI-safe, no network, no h5py):
      * retrieve_aster_ged_bands scales DN → emissivity (scale 0.001)
      * fill DN (-1) → nan bands → nan BBE
      * no-tile-found → nan
      * cache reuse: second call with same site hits the cache, no re-search
      * AsterGEDEmissivity BBE hand-computed (Ogawa 2004 ASTER coeffs)
      * EmissivitySource protocol conformance
      * tile-name predictor
  - Credentials helper: config.earthdata_credentials() raises without env
    (re-used guard, ensures no plaintext creds in source)
  - Constraint: zero system GDAL (no osgeo/gdal/pymodis) in the source
  - Real golden (gated on EARTHDATA_USERNAME): psu AG100 BBE ≈ 0.972 (Cheng & Liang 2014)

The original Plan 1b brief named the AppEEARS point API; that was verified
unavailable for ASTER GED during implementation (see module docstring in
``src/lstnet/io/aster_ged.py``). The working path is CMR + HDF5 (h5py), which
these tests exercise with mocked seams plus a gated real-retrieval test.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lstnet import config
from lstnet.io.aster_ged import (
    AsterGEDEmissivity,
    _OGAWA_COEFFS,
    _predicted_tile_filename,
    aster_to_broadband,
    retrieve_aster_ged_bands,
)
from lstnet.io.base import EmissivitySource
from lstnet.models import Site

_PSU = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
_TIME = datetime(2022, 7, 15, 14, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fake seams — make the full chain runnable with no network and no h5py.
# ---------------------------------------------------------------------------

class _FakeSearcher:
    """Returns a fixed HTTPS granule URL; ignores inputs."""

    def __init__(self, url="https://example/AG100.v003.41.-078.0001/AG100.v003.41.-078.0001.h5"):
        self.url = url
        self.calls = []

    def __call__(self, short_name, version, lon, lat):
        self.calls.append((short_name, version, lon, lat))
        return self.url


class _FakeDownloader:
    """Writes a non-empty placeholder file so the cache path exists.

    The size-based cache guard rejects zero-byte partial downloads, so the
    placeholder must be non-empty.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        p = dest_dir / name
        p.write_bytes(b"placeholder-h5-content")
        self.calls.append((url, p))
        return p


def _fake_reader(dn10, dn11, dn12, dn13, dn14):
    """Reader that ignores (lon,lat) and returns fixed 5-band DNs."""

    def _reader(path, lon, lat):
        return (int(dn10), int(dn11), int(dn12), int(dn13), int(dn14))

    return _reader


def _run_retrieve(reader, tmp_path, site=_PSU, **kw):
    """Drive retrieve_aster_ged_bands with the fake seams."""
    return retrieve_aster_ged_bands(
        site, data_dir=tmp_path / "ASTER_GED",
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=reader, **kw,
    )


# ---------------------------------------------------------------------------
# Scaling + OK path
# ---------------------------------------------------------------------------

def test_retrieve_scales_dn_to_emissivity(tmp_path):
    # psu real values: DN [970, 969, 966, 974, 974] → 0.970/0.969/0.966/0.974/0.974
    bands = _run_retrieve(_fake_reader(970, 969, 966, 974, 974), tmp_path)
    assert bands == pytest.approx((0.970, 0.969, 0.966, 0.974, 0.974), abs=1e-6)


def test_retrieve_uses_cache_avoids_research(tmp_path):
    """Second call with same site hits the cache — searcher not invoked."""
    searcher = _FakeSearcher()
    retrieve_aster_ged_bands(
        _PSU, data_dir=tmp_path / "ASTER_GED",
        searcher=searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    assert len(searcher.calls) == 1
    retrieve_aster_ged_bands(
        _PSU, data_dir=tmp_path / "ASTER_GED",
        searcher=searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    assert len(searcher.calls) == 1  # cache hit, no re-search


def test_retrieve_no_tile_returns_nan(tmp_path):
    def _none_searcher(short_name, version, lon, lat):
        return None

    bands = retrieve_aster_ged_bands(
        _PSU, data_dir=tmp_path / "ASTER_GED",
        searcher=_none_searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    assert all(math.isnan(b) for b in bands)


def test_retrieve_fill_dn_returns_nan(tmp_path):
    """DN == -1 is the ASTER GED fill value (water / no observations)."""
    bands = _run_retrieve(_fake_reader(-1, 969, 966, 974, 974), tmp_path)
    assert all(math.isnan(b) for b in bands)


# ---------------------------------------------------------------------------
# Broadband conversion (Ogawa 2004 ASTER)
# ---------------------------------------------------------------------------

def test_aster_to_broadband_hand_computed():
    # bands 10-14 = 0.970/0.969/0.966/0.974/0.974 (psu AG100 real values)
    # Default (Cheng & Liang 2014, 5-band, all weighted):
    #   0.197 + 0.025*0.970 + 0.057*0.969 + 0.237*0.966 + 0.333*0.974 + 0.146*0.974 = 0.971971
    bbe = aster_to_broadband(0.970, 0.969, 0.966, 0.974, 0.974)
    assert bbe == pytest.approx(0.971971, abs=1e-5)
    # Ogawa 3-band alternative (secondary-sourced): 0.127*0.970 + 0.711*0.969 + 0.162*0.966 = 0.968641
    bbe_ogawa = aster_to_broadband(0.970, 0.969, 0.966, 0.974, 0.974, bbe="ogawa")
    assert bbe_ogawa == pytest.approx(0.968641, abs=1e-5)


def test_aster_to_broadband_nan_propagates():
    assert math.isnan(aster_to_broadband(float("nan"), 0.969, 0.966, 0.974, 0.974))
    assert math.isnan(aster_to_broadband(0.970, float("nan"), 0.966, 0.974, 0.974))


def test_ogawa_coeffs_sum_to_one():
    """Ogawa 2004 ASTER coefficients should sum to ~1 (weighted-mean property)."""
    assert sum(_OGAWA_COEFFS) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# AsterGEDEmissivity BBE + protocol
# ---------------------------------------------------------------------------

def test_aster_ged_satisfies_emissivity_source_protocol():
    assert isinstance(AsterGEDEmissivity(), EmissivitySource)
    assert AsterGEDEmissivity().name == "aster_ged"


def test_aster_ged_bbe_hand_computed():
    # DN [970, 969, 966, 974, 974] → BBE 0.971971 (Cheng & Liang 2014, default)
    src = AsterGEDEmissivity(
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    bbe = src.emissivity(_PSU, _TIME, "Day")
    assert bbe == pytest.approx(0.971971, abs=1e-5)


def test_aster_ged_overpass_time_unused(tmp_path):
    """AG100 is a 2000–2008 mean — BBE is the same for any overpass moment."""
    src = AsterGEDEmissivity(
        data_dir=tmp_path,
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    bbe_day = src.emissivity(_PSU, _TIME, "Day")
    bbe_night = src.emissivity(_PSU, _TIME.replace(hour=4), "Night")
    assert bbe_day == bbe_night


def test_aster_ged_fill_pixel_returns_nan(tmp_path):
    src = AsterGEDEmissivity(
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(-1, 969, 966, 974, 974),
    )
    assert math.isnan(src.emissivity(_PSU, _TIME, "Day"))


def test_aster_ged_no_tile_returns_nan(tmp_path):
    def _none(short_name, version, lon, lat):
        return None
    # Pass data_dir so the test does not read a project-default cache that may
    # have been populated by the gated real-retrieval test or a prior run.
    src = AsterGEDEmissivity(
        data_dir=tmp_path / "ASTER_GED",
        searcher=_none, downloader=_FakeDownloader(),
        reader=_fake_reader(970, 969, 966, 974, 974),
    )
    assert math.isnan(src.emissivity(_PSU, _TIME, "Day"))


# ---------------------------------------------------------------------------
# Tile-name predictor (cache-hit hint)
# ---------------------------------------------------------------------------

def test_predicted_tile_filename_psu():
    # psu (-77.93, 40.72) → tile "AG100.v003.41.-078" (covers lon [-78,-77), lat [40,41))
    # Verified against the real downloaded granule id.
    name = _predicted_tile_filename(-77.93, 40.72)
    assert name == "AG100.v003.41.-078"


def test_predicted_tile_filename_eastern_hemisphere():
    # A HiWATER site (arz) at (100.46, 38.05) → tile lat 39, lon 100
    name = _predicted_tile_filename(100.46, 38.05)
    assert name == "AG100.v003.39.100"


def test_predicted_tile_filename_southern_hemisphere():
    # A southern-hemisphere site: lat -38.05 → ceil = -38; lon 100.46 → floor 100
    name = _predicted_tile_filename(100.46, -38.05)
    assert name == "AG100.v003.-38.100"


# ---------------------------------------------------------------------------
# Credentials helper (env-only; never hardcoded)
# ---------------------------------------------------------------------------

def test_earthdata_credentials_raises_without_env(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        config.earthdata_credentials()


def test_no_hardcoded_legacy_creds_in_aster_source():
    """Credentials must come from env only — grep guard against regressions."""
    root = Path(__file__).resolve().parents[1] / "src" / "lstnet"
    banned = ("yaoganlou", "1990Ts", "REDACTED_USER", "REDACTED_EMAIL")
    hits = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                hits.append((str(p), token))
    assert hits == [], f"hardcoded credential tokens found: {hits}"


def test_zero_system_gdal_in_aster_source():
    """Constraint: h5py only; no osgeo/gdal/pymodis *imports* in the source."""
    src = Path(__file__).resolve().parents[1] / "src" / "lstnet" / "io" / "aster_ged.py"
    text = src.read_text(encoding="utf-8")
    import_lines = [
        ln.strip() for ln in text.splitlines()
        if ln.lstrip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    assert "from osgeo" not in joined
    assert "import gdal" not in joined
    assert "import pymodis" not in joined
    assert "from pymodis" not in joined


# ---------------------------------------------------------------------------
# Real golden (gated on credentials + network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("EARTHDATA_USERNAME"),
    reason="Needs EARTHDATA_USERNAME/PASSWORD + network; uses real NASA CMR + LP DAAC.",
)
def test_golden_psu_aster_ged_bbe(tmp_path):
    """Real retrieval: psu AG100 V003 → BBE ≈ 0.972 (Cheng & Liang 2014, default).

    Verified end-to-end during implementation: DN [970,969,966,974,974] at the
    pixel nearest (-77.93, 40.72); 16 observations, NDVI 0.54 (vegetated, in
    the expected 0.97–0.99 BBE range). Uses the live CMR+h5py path.
    """
    src = AsterGEDEmissivity(data_dir=tmp_path / "ASTER_GED")
    bbe = src.emissivity(_PSU, _TIME, "Day")
    assert bbe == pytest.approx(0.972, abs=0.005)
