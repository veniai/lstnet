"""Tests for the SURFRAD reader (Task 8).

Covers: G4 (fixture parse + missing-file boundary), G11 (HTTPS not FTP),
G13 (golden sample parity with legacy SURFRADlst), G14 (pre-1995 OutOfDate).

The reader is exercised fully offline via ``data_dir`` pointing at the
committed fixture; ``requests`` is never called in these tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lstnet.ground_lst import lst_from_radiance
from lstnet.io.base import NetworkReader, RadiationWindow
from lstnet.io.surfrad import SURFRAD_BASE_URL, SurfradReader
from lstnet.models import Site
from lstnet.qc import decide_qc

FIXTURES = Path(__file__).parent / "fixtures"

# Bondville, 2011-02-12 14:30 UTC — inside data/SURFRAD/bon11043.dat row 872.
_BON = Site(name="bon", network="SURFRAD", lon=-88.37, lat=40.05)
_OVERPASS = datetime(2011, 2, 12, 14, 30, tzinfo=timezone.utc)
_WINDOW = 4  # ±4 min -> matches legacy step=1 range(row-4, row+5) = 9 samples


def _reader() -> SurfradReader:
    return SurfradReader(data_dir=FIXTURES)


def _offline_reader() -> SurfradReader:
    """Reader whose cache-miss path never touches the network."""
    def _no_fetch(url: str, dest: Path) -> None:
        raise FileNotFoundError(url)
    return SurfradReader(data_dir=FIXTURES, fetcher=_no_fetch)


def test_satisfies_network_reader_protocol():
    assert isinstance(SurfradReader(data_dir=FIXTURES), NetworkReader)


def test_parses_fixture_known_overpass_ok_status():
    win = _reader().read_radiation(_BON, _OVERPASS, _WINDOW)
    assert isinstance(win, RadiationWindow)
    assert win.status == "OK"
    # Legacy range(row-4, row+5) for step=1 = 9 samples (14:26..14:34).
    assert len(win.samples) == 9
    # Overpass row (14:30) is the middle sample; assert its exact L_down/L_up.
    mid = win.samples[len(win.samples) // 2]
    assert mid.time == _OVERPASS
    assert mid.l_down == pytest.approx(214.9)
    assert mid.l_up == pytest.approx(292.0)


def test_missing_file_returns_file_not_found_status():
    nowhere = Site(name="bon", network="SURFRAD", lon=0.0, lat=0.0)
    # An overpass on a day with no committed file; stubbed fetcher keeps it
    # fully offline (no real network call).
    win = _offline_reader().read_radiation(
        nowhere, datetime(2011, 2, 13, 14, 30, tzinfo=timezone.utc), _WINDOW
    )
    assert win.status == "FileNotFound"
    assert win.samples == []


def test_pre_1995_overpass_is_out_of_date():
    win = _reader().read_radiation(
        _BON, datetime(1994, 12, 31, 14, 30, tzinfo=timezone.utc), _WINDOW
    )
    assert win.status == "OutOfDate"
    assert win.samples == []


def test_base_url_is_https_not_ftp():
    # G11: HTTPS scheme, no FTP scheme, no legacy FTP host.
    assert SURFRAD_BASE_URL == "https://gml.noaa.gov/aftp/data/radiation/surfrad/"
    src = Path(__file__).resolve().parents[1] / "src" / "lstnet" / "io" / "surfrad.py"
    text = src.read_text(encoding="utf-8")
    assert "https://gml.noaa.gov/aftp/data/radiation/surfrad/" in text
    assert "ftp://" not in text
    assert "aftp.cmdl.noaa.gov" not in text
    assert "urllib.request" not in text  # legacy used urllib for FTP; we use requests


def test_golden_sample_matches_legacy_surfradlst():
    """G13: reproduce legacy SURFRADlst('bon','201102121430',0.95) == 268.82 K.

    Legacy used sigma=5.67e-8; new pipeline uses 5.670374e-8 → delta ~0.004 K,
    well inside the 0.01 K tolerance.
    """
    GOLDEN_LEGACY_LST_K = 268.82  # captured from methods/site_LST.py::SURFRADlst

    win = _reader().read_radiation(_BON, _OVERPASS, _WINDOW)
    assert win.status == "OK"

    samples_k = [
        lst_from_radiance(s.l_up, s.l_down, 0.95) for s in win.samples
    ]
    avg_k, qc = decide_qc(samples_k)
    assert qc == "OK"
    assert avg_k is not None
    assert avg_k == pytest.approx(GOLDEN_LEGACY_LST_K, abs=0.01)
