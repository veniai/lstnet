"""Tests for ``compute_ground_lst`` orchestration (G2, G14) and the G9 fix.

G2 — fake reader drives the full reader + physics + QC pipeline and yields the
     expected averaged LST + ``OK`` (and the std>1 / empty branches).
G14 — an unparseable overpass-time string returns ``TimeError`` instead of raising.
G9  — reader default ``data_dir`` is CWD-independent and absolute.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pytest

from lstnet import config
from lstnet.ground_lst import (
    SIGMA,
    compute_ground_lst,
    lst_from_radiance,
    parse_overpass_time,
)
from lstnet.io.base import NetworkReader, RadiationSample, RadiationWindow
from lstnet.io.emissivity import FixedEmissivity
from lstnet.io.hiwater import HiwaterReader
from lstnet.io.pku import PkuReader
from lstnet.io.surfrad import SurfradReader
from lstnet.models import GroundLST, Site
from lstnet.qc import QC_NO_DATA, QC_OK, QC_STD_ERROR, QC_TIME_ERROR

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
_TIME = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)
_EMISS = FixedEmissivity(0.95)


class _FakeReader:
    """Minimal ``NetworkReader`` returning a canned ``RadiationWindow``."""

    network = "FAKE"

    def __init__(self, window: RadiationWindow):
        self._window = window

    def read_radiation(self, site, overpass_time, window_minutes):
        return self._window


def _sample(l_up: float, l_down: float, t: datetime = _TIME) -> RadiationSample:
    return RadiationSample(time=t, l_up=l_up, l_down=l_down)


def _expected_lst(l_up: float, l_down: float, emiss: float = 0.95) -> float:
    return lst_from_radiance(l_up, l_down, emiss)


# --- G2: orchestration end-to-end with a fake reader -------------------------


def test_g2_ok_window_averages_samples():
    # Two near-identical samples (std well under 1 K) → OK + averaged LST.
    samples = [
        _sample(400.0, 300.0),
        _sample(400.0, 300.0),
    ]
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=samples, status=QC_OK
    )
    result = compute_ground_lst(_SITE, _TIME, _EMISS, _FakeReader(window))

    assert isinstance(result, GroundLST)
    expected = _expected_lst(400.0, 300.0)
    assert result.lst_k == pytest.approx(expected, rel=1e-9)
    assert result.qc_flag == QC_OK
    assert result.emissivity == pytest.approx(0.95)
    assert result.day_or_night == "Day"  # _TIME 2020-07-14 14:13 UTC at _SITE -> Day
    assert result.overpass_time == _TIME
    assert result.site is _SITE


def test_g2_std_above_threshold_yields_std_error():
    # Two samples whose LSTs differ by >~1.4 K → std>1 → StdError, lst_k=nan.
    # Pick l_up values so the two LSTs land ~5 K apart.
    l_up_a = 400.0
    # Solve for the l_up that raises LST by ~5 K from the baseline.
    base_t = _expected_lst(l_up_a, 300.0)
    raised_t = base_t + 5.0
    l_up_b = 0.95 * SIGMA * raised_t**4 + 0.05 * 300.0
    samples = [_sample(l_up_a, 300.0), _sample(l_up_b, 300.0)]
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=samples, status=QC_OK
    )
    result = compute_ground_lst(_SITE, _TIME, _EMISS, _FakeReader(window))

    assert result.qc_flag == QC_STD_ERROR
    assert math.isnan(result.lst_k)


def test_g2_empty_ok_window_yields_no_data():
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=[], status=QC_OK
    )
    result = compute_ground_lst(_SITE, _TIME, _EMISS, _FakeReader(window))

    assert result.qc_flag == QC_NO_DATA
    assert math.isnan(result.lst_k)


def test_g2_non_ok_window_propagates_status():
    for status in ("NoData", "FileNotFound", "DataError", "OutOfDate"):
        window = RadiationWindow(
            site=_SITE, overpass_time=_TIME, samples=[], status=status
        )
        result = compute_ground_lst(_SITE, _TIME, _EMISS, _FakeReader(window))
        assert result.qc_flag == status, status
        assert math.isnan(result.lst_k)
        # Emissivity is still resolved even when the window fails.
        assert result.emissivity == pytest.approx(0.95)


def test_g2_non_physical_samples_are_skipped():
    # First sample non-physical (l_up < (1-emiss)*l_down = 15), second valid.
    non_physical = _sample(10.0, 300.0)   # 10 < 15 → ValueError in physics
    valid = _sample(400.0, 300.0)
    window = RadiationWindow(
        site=_SITE,
        overpass_time=_TIME,
        samples=[non_physical, valid],
        status=QC_OK,
    )
    # Only one valid sample → decide_qc returns NoData (<2 samples).
    result = compute_ground_lst(_SITE, _TIME, _EMISS, _FakeReader(window))
    assert result.qc_flag == QC_NO_DATA


def test_g2_str_overpass_time_is_parsed():
    samples = [_sample(400.0, 300.0), _sample(400.0, 300.0)]
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=samples, status=QC_OK
    )
    result = compute_ground_lst(
        _SITE, "202007141413", _EMISS, _FakeReader(window)
    )
    assert result.qc_flag == QC_OK
    assert result.overpass_time == _TIME


# --- G14: unparseable overpass time → TimeError (no exception) ---------------


def test_g14_unparseable_overpass_time_returns_time_error():
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=[], status=QC_OK
    )
    result = compute_ground_lst(
        _SITE, "not-a-time", _EMISS, _FakeReader(window)
    )
    assert result.qc_flag == QC_TIME_ERROR
    assert math.isnan(result.lst_k)
    assert math.isnan(result.emissivity)
    assert result.day_or_night == "Unknown"
    # Sentinel overpass_time for the TimeError path.
    assert result.overpass_time == datetime(1, 1, 1, tzinfo=timezone.utc)


def test_g14_wrong_length_digit_string_returns_time_error():
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=[], status=QC_OK
    )
    # 11 digits, not 12.
    result = compute_ground_lst(
        _SITE, "20200714141", _EMISS, _FakeReader(window)
    )
    assert result.qc_flag == QC_TIME_ERROR


# --- parse_overpass_time unit checks -----------------------------------------


def test_parse_overpass_time_valid():
    assert parse_overpass_time("202007141413") == _TIME


def test_parse_overpass_time_invalid_raises():
    for bad in ("not-a-time", "20200714141", "202013141413", "", None, 202007141413):
        with pytest.raises(ValueError):
            parse_overpass_time(bad)


# --- G9: reader defaults are CWD-independent absolute paths ------------------


def test_g9_surfrad_default_data_dir_is_absolute_and_under_project_root():
    reader = SurfradReader()
    assert reader.data_dir.is_absolute()
    assert reader.data_dir == config.project_root() / "data" / "SURFRAD"


def test_g9_pku_default_data_dir_is_absolute_and_under_project_root():
    reader = PkuReader()
    assert reader.data_dir.is_absolute()
    assert reader.data_dir == config.project_root() / "data" / "pku-sites"


def test_g9_hiwater_default_data_dir_is_absolute_and_under_project_root():
    reader = HiwaterReader()
    assert reader.data_dir.is_absolute()
    assert reader.data_dir == config.project_root() / "data" / "HiWATER"


def test_g9_defaults_independent_of_cwd(tmp_path, monkeypatch):
    # Constructing a reader from a different CWD must yield the same path.
    monkeypatch.chdir(tmp_path)
    expected = config.project_root() / "data" / "SURFRAD"
    assert SurfradReader().data_dir == expected
    assert os.getcwd() != str(config.project_root())


def test_g9_no_getcwd_in_src():
    # No reader or config code reads the CWD.
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-rnE",
            r"getcwd|os\.getcwd",
            str(config.package_root()),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"unexpected getcwd usage:\n{result.stdout}"
    assert result.stdout == ""


def test_fake_reader_satisfies_protocol():
    window = RadiationWindow(
        site=_SITE, overpass_time=_TIME, samples=[], status=QC_OK
    )
    assert isinstance(_FakeReader(window), NetworkReader)
