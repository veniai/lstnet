"""Tests for the MODIS MYD21A1D/A1N C6.1 daily emissivity source (Plan 1b-T3).

Coverage:
  - Mocked (CI-safe, no network, no pyhdf):
      * tile geometry (psu → h12v04)
      * retrieve_emissivity_bands scales DN → emissivity and reports OK QC
      * QC masking: cloud / not-produced / missing-pixel → 'Missing'
      * padding guard: fill DN → 'Fill'; out-of-tile pixel → 'Fill'
      * ModisDailyEmissivity BBE (modern + legacy) hand-computed
      * EmissivitySource protocol conformance
      * short_name dispatch Day/Night
  - Credentials helper: config.earthdata_credentials() raises without env
  - Real golden (gated on EARTHDATA_USERNAME): psu 2022-07-15 day BBE ≈ 0.9652
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lstnet import config
from lstnet.io.base import EmissivitySource
from lstnet.io.emissivity import (
    ModisDailyEmissivity,
    broadband_emissivity,
)
from lstnet.io.modis import (
    EmissivityBands,
    _EMIS_OFFSET,
    _EMIS_SCALE,
    modis_tile,
    retrieve_emissivity_bands,
)
from lstnet.models import Site

# psu (the spike's reference site): lon -77.93, lat 40.72, MODIS tile h12v04.
_PSU = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
# 2022-07-15 = A2022196 — the granule verified end-to-end in the spike.
_PSU_DAY = datetime(2022, 7, 15, 14, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------

def test_modis_tile_psu_is_h12v04():
    """psu (lon -77.93, lat 40.72) maps to MODIS sinusoidal tile h12v04."""
    h, v = modis_tile(_PSU)
    assert (h, v) == (12, 4)


def test_modis_tile_clamps_to_global_grid():
    # High-latitude / antimeridian sites should not raise or escape the grid.
    s = Site(name="x", network="t", lon=179.9, lat=80.0)
    h, v = modis_tile(s)
    assert 0 <= h <= 35
    assert 0 <= v <= 17


# ---------------------------------------------------------------------------
# Fake seams — make the full chain runnable with no network and no pyhdf.
# ---------------------------------------------------------------------------

class _FakeSearcher:
    """Returns a fixed HTTPS granule URL; ignores inputs."""

    def __init__(self, url="https://example/MYD21A1D.A2022196.h12v04.061.0000.hdf"):
        self.url = url
        self.calls = []

    def __call__(self, short_name, version, tile, date):
        self.calls.append((short_name, version, tile, date))
        return self.url


class _FakeDownloader:
    """Writes a placeholder file large enough to pass the cache-validity guard.

    Real MODIS granules are 100 KB+; the guard rejects files below
    ``_MIN_CACHED_BYTES``, so the fake writes a body well above that so a
    second call hits the cache (no re-search).
    """

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        p = dest_dir / name
        p.write_bytes(b"placeholder-hdf-" * 1000)  # ~16 KB > _MIN_CACHED_BYTES
        self.calls.append((url, p))
        return p


def _fake_reader(dn29, dn31, dn32, qc=0):
    """Reader that ignores ``(row,col)`` and returns fixed DNs + QC."""

    def _reader(path, row, col):
        return int(dn29), int(dn31), int(dn32), int(qc)

    return _reader


def _run_retrieve(reader, tmp_path, site=_PSU, day_or_night="Day", **kw):
    """Drive retrieve_emissivity_bands with the fake seams."""
    return retrieve_emissivity_bands(
        site, _PSU_DAY, day_or_night, data_dir=tmp_path / "MODIS",
        searcher=_FakeSearcher(),
        downloader=_FakeDownloader(),
        reader=reader,
        **kw,
    )


# ---------------------------------------------------------------------------
# Scaling + OK path
# ---------------------------------------------------------------------------

def test_retrieve_scales_dn_to_emissivity_ok_qc(tmp_path):
    # Spike values: dn29=229, dn31=243, dn32=242 → 0.9480/0.9760/0.9740.
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=0), tmp_path)
    assert bands.qc == "OK"
    assert bands.em29 == pytest.approx(0.9480, abs=1e-4)
    assert bands.em31 == pytest.approx(0.9760, abs=1e-4)
    assert bands.em32 == pytest.approx(0.9740, abs=1e-4)


def test_retrieve_uses_cache_avoids_research(tmp_path):
    """Second call with same date+tile hits the cache — searcher not invoked."""
    searcher = _FakeSearcher()
    retrieve_emissivity_bands(
        _PSU, _PSU_DAY, "Day", data_dir=tmp_path / "MODIS",
        searcher=searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242),
    )
    assert len(searcher.calls) == 1
    # Second call: downloader output is already cached under data_dir.
    retrieve_emissivity_bands(
        _PSU, _PSU_DAY, "Day", data_dir=tmp_path / "MODIS",
        searcher=searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242),
    )
    assert len(searcher.calls) == 1  # cache hit, no re-search


def test_retrieve_rejects_too_small_cached_file(tmp_path):
    """A tiny cached file (a leaked test placeholder or a failed partial) is
    treated as a cache miss and re-downloaded, not reused as a bogus HDF
    (which would crash pyhdf). Guards the cache-poisoning bug."""
    cache = tmp_path / "MODIS"
    cache.mkdir()
    bogus = cache / "MYD21A1D.A2022196.h12v04.061.0000.hdf"
    bogus.write_bytes(b"placeholder-hdf-content")  # 23 bytes < _MIN_CACHED_BYTES
    downloader = _FakeDownloader()
    bands = retrieve_emissivity_bands(
        _PSU, _PSU_DAY, "Day", data_dir=cache,
        searcher=_FakeSearcher(), downloader=downloader,
        reader=_fake_reader(229, 243, 242),
    )
    assert bands.qc == "OK"               # re-downloaded + read fine
    assert len(downloader.calls) == 1     # bogus file rejected -> cache miss


def test_retrieve_no_granule_returns_missing(tmp_path):
    def _none_searcher(short_name, version, tile, date):
        return None

    bands = retrieve_emissivity_bands(
        _PSU, _PSU_DAY, "Day", data_dir=tmp_path / "MODIS",
        searcher=_none_searcher, downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242),
    )
    assert bands.qc == "Missing"
    assert math.isnan(bands.em29)


# ---------------------------------------------------------------------------
# QC masking (legacy ignored QC entirely)
# ---------------------------------------------------------------------------

def test_qc_cloud_pixel_marked_missing(tmp_path):
    # MODLAND_QA bits [0-1] = 10 (cloud) → pixel not produced.
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=0b10), tmp_path)
    assert bands.qc == "Missing"
    assert math.isnan(bands.em31)


def test_qc_other_reason_marked_missing(tmp_path):
    # bits [0-1] = 11 (other reason) → not produced.
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=0b11), tmp_path)
    assert bands.qc == "Missing"


def test_qc_data_quality_missing_pixel_marked_missing(tmp_path):
    # bits [2-3] = 01 (Missing Pixel) → 0b0100.
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=0b0100), tmp_path)
    assert bands.qc == "Missing"


def test_qc_unreliable_quality_still_accepted(tmp_path):
    """MODLAND_QA=01 means 'produced, unreliable' — still a usable retrieval."""
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=0b01), tmp_path)
    assert bands.qc == "OK"
    assert bands.em31 == pytest.approx(0.9760, abs=1e-4)


def test_qc_actual_psu_word_accepted(tmp_path):
    """The real psu QC word from the spike HDF (48865) is QA=01 (unreliable but produced)."""
    bands = _run_retrieve(_fake_reader(229, 243, 242, qc=48865), tmp_path)
    assert bands.qc == "OK"


# ---------------------------------------------------------------------------
# Padding guard (legacy crashed at tile edges)
# ---------------------------------------------------------------------------

def test_fill_dn_marked_fill(tmp_path):
    """DN=0 is the MODIS fill value (padded pixels outside the swath)."""
    bands = _run_retrieve(_fake_reader(0, 243, 242), tmp_path)
    assert bands.qc == "Fill"
    assert math.isnan(bands.em29)


def test_out_of_tile_pixel_marked_fill(tmp_path, monkeypatch):
    """A site whose sinusoidal pixel falls outside [0,1200) is padding, not data.

    Geometry floor-division keeps real-world sites inside their tile, so this
    simulates the legacy edge condition directly: monkeypatch the pixel lookup
    to return an out-of-tile index and assert the guard fires instead of
    indexing into padding.
    """
    import lstnet.io.modis as modis_mod

    monkeypatch.setattr(modis_mod, "_pixel_index", lambda site, ux, uy: (1113, 1200))
    bands = _run_retrieve(_fake_reader(229, 243, 242), tmp_path)
    assert bands.qc == "Fill"
    assert math.isnan(bands.em29)


def test_invalid_dn_marked_missing(tmp_path):
    """A DN outside [1,255] (but not 0) is treated as a corrupt/missing retrieval."""
    bands = _run_retrieve(_fake_reader(229, 300, 242), tmp_path)
    assert bands.qc == "Missing"


# ---------------------------------------------------------------------------
# ModisDailyEmissivity BBE + protocol
# ---------------------------------------------------------------------------

def test_modis_daily_satisfies_emissivity_source_protocol():
    assert isinstance(ModisDailyEmissivity(), EmissivitySource)
    assert ModisDailyEmissivity().name == "modis_daily"


def test_modis_daily_modern_bbe_hand_computed():
    # dn29=229, dn31=243, dn32=242 → 0.9480/0.9760/0.9740
    # modern = 0.047*0.948 + 0.977*0.976 - 0.024*0.974 = 0.974732 (Ogawa 2004)
    src = ModisDailyEmissivity(
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242),
    )
    bbe = src.emissivity(_PSU, _PSU_DAY, "Day")
    assert bbe == pytest.approx(0.974732, abs=1e-4)


def test_modis_daily_legacy_bbe_matches_spike():
    # legacy bare-soil: 0.329*0.948 + 0.572*0.976 + 0.095 = 0.9652 (spike result)
    src = ModisDailyEmissivity(
        bbe="legacy",
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242),
    )
    assert src.emissivity(_PSU, _PSU_DAY, "Day") == pytest.approx(0.9652, abs=1e-4)


def test_modis_daily_qc_missing_returns_nan():
    src = ModisDailyEmissivity(
        searcher=_FakeSearcher(), downloader=_FakeDownloader(),
        reader=_fake_reader(229, 243, 242, qc=0b10),  # cloud
    )
    assert math.isnan(src.emissivity(_PSU, _PSU_DAY, "Day"))


def test_modis_daily_invalid_bbe_raises():
    with pytest.raises(ValueError):
        ModisDailyEmissivity(bbe="bogus")


def test_short_name_day_night_dispatch():
    from lstnet.io.modis import _short_name_for
    assert _short_name_for("Day") == "MYD21A1D"
    assert _short_name_for("Night") == "MYD21A1N"
    assert _short_name_for("night") == "MYD21A1N"
    with pytest.raises(ValueError):
        _short_name_for("dusk")


def test_broadband_emissivity_helpers():
    ok = EmissivityBands(0.948, 0.976, 0.974, "OK")
    assert broadband_emissivity(ok, "modern") == pytest.approx(0.974732, abs=1e-4)
    assert broadband_emissivity(ok, "legacy") == pytest.approx(0.9652, abs=1e-4)
    bad = EmissivityBands(float("nan"), float("nan"), float("nan"), "Missing")
    assert math.isnan(broadband_emissivity(bad, "modern"))


# ---------------------------------------------------------------------------
# Credentials helper (env-only; never hardcoded)
# ---------------------------------------------------------------------------

def test_earthdata_credentials_raises_without_env(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        config.earthdata_credentials()


def test_earthdata_credentials_returns_pair_when_set(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    assert config.earthdata_credentials() == ("u", "p")


def test_no_hardcoded_legacy_creds_in_source():
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


def test_zero_system_gdal_in_modis_source():
    """Constraint: pyhdf only; no osgeo/gdal/pymodis *imports* in the source."""
    src = Path(__file__).resolve().parents[1] / "src" / "lstnet" / "io" / "modis.py"
    text = src.read_text(encoding="utf-8")
    # Match import statements only (docstrings may mention pymodis as a contrast).
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
    reason="Needs EARTHDATA_USERNAME/PASSWORD + network; uses real NASA CMR.",
)
def test_golden_psu_modis_bbe_matches_spike(tmp_path):
    """Real retrieval: psu MYD21A1D.061 for 2022-07-15 day → legacy BBE ≈ 0.9652.

    Spike report (.git/sdd/spike-1b-report.md) obtained em29=0.9480 /
    em31=0.9760 / em32=0.9740 → legacy bare-soil BBE 0.9652. This test exercises
    the live earthaccess+pyhdf path with the default seams.
    """
    src = ModisDailyEmissivity(data_dir=tmp_path / "MODIS", bbe="legacy")
    bbe = src.emissivity(_PSU, _PSU_DAY, "Day")
    assert bbe == pytest.approx(0.9652, abs=0.01)


def test_strict_qc_rejects_unreliable_pixel():
    """strict=True rejects MODLAND_QA=01 (unreliable); lenient accepts it."""
    from lstnet.io.modis import _scale_and_qc

    # dn valid (200 → 0.89); qc_word=1 means MODLAND_QA bits = 01 (unreliable, produced)
    lenient = _scale_and_qc(200, 210, 215, 1)
    assert lenient.qc == "OK"
    strict = _scale_and_qc(200, 210, 215, 1, strict=True)
    assert strict.qc == "Missing"
    # QA=00 is always OK, even in strict mode
    assert _scale_and_qc(200, 210, 215, 0, strict=True).qc == "OK"
