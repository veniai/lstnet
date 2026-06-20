"""Tests for the PKULSTNet reader (Task 9).

Covers: G5 (offline fixture parse, hbc swap, FileNotFound, DataError, golden
parity with legacy ``methods/site_LST.py::PKULSTNet``).

The reader is exercised fully offline via ``data_dir`` pointing at the
committed fixture; no network is involved (PKU data is local-only).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lstnet.ground_lst import SIGMA, lst_from_radiance
from lstnet.io.base import NetworkReader, RadiationWindow
from lstnet.io.pku import (
    PkuReader,
    QC_DATA_ERROR,
    QC_FILE_NOT_FOUND,
    QC_OK,
    brightness_to_radiance,
)
from lstnet.models import Site
from lstnet.qc import decide_qc

FIXTURES = Path(__file__).parent / "fixtures"

# 承德站 (hbc), overpass 2018-08-09 18:30 UTC → +8 h → 2018-08-10 02:30 Beijing.
# The fixture is a 41-row slice of the production file centred on that time;
# rows are 3-min apart, so the 3-min branch fires with window data[a-1:b+2].
_HBC = Site(name="hbc", network="PKULSTNet", lon=117.24719, lat=42.41165)
_OVERPASS_UTC = datetime(2018, 8, 9, 18, 30)
_WINDOW = 0  # ignored by PkuReader — interval-branch decides the window

# Legacy captured value (methods/site_LST.py::PKULSTNet('hbc','201808091830',0.95))
# via py3-compatible re-run; the legacy .2f result and the raw mean both stored.
_GOLDEN_LEGACY_LST_K = 285.81
_GOLDEN_LEGACY_LST_RAW = 285.80735889601607


def _reader() -> PkuReader:
    return PkuReader(data_dir=FIXTURES / "pku-sites")


def test_satisfies_network_reader_protocol():
    assert isinstance(PkuReader(data_dir=FIXTURES), NetworkReader)


def test_known_overpass_returns_ok_three_samples_3min_branch():
    """3-min interval → window data[a-1:b+2] = exactly 3 samples."""
    win = _reader().read_radiation(_HBC, _OVERPASS_UTC, _WINDOW)
    assert isinstance(win, RadiationWindow)
    assert win.status == QC_OK
    assert len(win.samples) == 3
    # Middle sample is the overpass row (02:30 Beijing = 18:30 UTC prev day).
    mid = win.samples[1]
    assert mid.time == _OVERPASS_UTC
    # Source row: TagTemp(1)=11.28751, TagTemp(2)=-20.34698 → t1=284.43751, t2=252.80302
    # hbc swap: t1 = TagTemp(1)+273.15
    expected_l_up = SIGMA * (11.28751 + 273.15) ** 4
    assert mid.l_up == pytest.approx(expected_l_up, rel=1e-12)


def test_hbc_swap_applied_before_radiance_conversion():
    """The hbc path must differ from the non-hbc path on the same row.

    Legacy _retrieval: hbc → t1=item[2], t2=item[3]; others → t1=item[3], t2=item[2].
    With identical brightness temperatures the swap is invisible, so use the
    real fixture row where TagTemp(1) != TagTemp(2).
    """
    reader = _reader()
    win = reader.read_radiation(_HBC, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_OK
    mid = win.samples[1]

    # Synthesize the SAME row as a plain array and run both code paths.
    fake_row = [None, None, 11.28751, -20.34698]  # indices 2,3 are TagTemp(1),(2)
    l_up_hbc, l_down_hbc = brightness_to_radiance(fake_row, "hbc")
    l_up_other, l_down_other = brightness_to_radiance(fake_row, "hnw")

    # hbc reader output matches the hbc direct call.
    assert mid.l_up == pytest.approx(l_up_hbc, rel=1e-12)
    assert mid.l_down == pytest.approx(l_down_hbc, rel=1e-12)
    # And the hbc path genuinely differs from the non-hbc path (swap is real).
    assert l_up_hbc != l_up_other
    assert l_down_hbc != l_down_other
    # The swap is symmetric: hbc's l_up == other's l_down and vice-versa.
    assert l_up_hbc == pytest.approx(l_down_other, rel=1e-12)
    assert l_down_hbc == pytest.approx(l_up_other, rel=1e-12)


def test_algebra_equivalent_radiance_matches_legacy_paired_formula():
    """lst_from_radiance(σ·t1⁴, σ·t2⁴, emiss) == legacy paired formula (σ cancels)."""
    t1, t2, emiss = 284.43751, 252.80302, 0.95
    legacy = ((t1**4 - (1 - emiss) * t2**4) / emiss) ** 0.25
    via_pipeline = lst_from_radiance(SIGMA * t1**4, SIGMA * t2**4, emiss)
    assert via_pipeline == pytest.approx(legacy, abs=1e-9)


def test_missing_file_returns_file_not_found_status():
    # A month with no committed fixture file.
    nowhere = Site(name="hbc", network="PKULSTNet", lon=0.0, lat=0.0)
    win = _reader().read_radiation(nowhere, datetime(2017, 1, 1, 0, 0), _WINDOW)
    assert win.status == QC_FILE_NOT_FOUND
    assert win.samples == []


def test_unknown_site_name_returns_file_not_found():
    """Site not in _SITE_CN → glob still runs with the raw name → no match."""
    unknown = Site(name="zzz", network="PKULSTNet", lon=0.0, lat=0.0)
    win = _reader().read_radiation(unknown, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_FILE_NOT_FOUND


def test_data_error_when_overpass_outside_window_threshold():
    """An overpass whose nearest rows fall outside the 9-min 3-min threshold.

    The fixture spans 01:30..03:30 Beijing; an overpass at 10:00 UTC+8 = 18:00
    Beijing is well outside the file's time range, so _nearest pins to the
    edge and the interval/threshold checks fail → DataError.
    """
    # 02:00 UTC + 8h = 10:00 Beijing — outside the 01:30..03:30 fixture span.
    far = datetime(2018, 8, 10, 2, 0)
    win = _reader().read_radiation(_HBC, far, _WINDOW)
    assert win.status == QC_DATA_ERROR
    assert win.samples == []


def test_golden_sample_matches_legacy_pkulstnet():
    """Reproduce legacy PKULSTNet('hbc','201808091830',0.95) == 285.81 K.

    Legacy returns ``format(avg, '.2f')``; raw mean 285.80736 K. The new
    pipeline (PkuReader → lst_from_radiance → decide_qc) must match within
    0.01 K. σ cancels in the algebra, so SIGMA choice is irrelevant here.
    """
    win = _reader().read_radiation(_HBC, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_OK

    samples_k = [lst_from_radiance(s.l_up, s.l_down, 0.95) for s in win.samples]
    avg_k, qc = decide_qc(samples_k)
    assert qc == "OK"
    assert avg_k is not None
    # Tolerance 0.01 K covers the .2f rounding of the legacy captured value.
    assert avg_k == pytest.approx(_GOLDEN_LEGACY_LST_K, abs=0.01)
    # And the unrounded mean matches the legacy raw mean essentially exactly.
    assert avg_k == pytest.approx(_GOLDEN_LEGACY_LST_RAW, abs=1e-6)
