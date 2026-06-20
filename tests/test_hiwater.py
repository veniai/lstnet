"""Tests for the HiWATER reader (Task 10).

Covers: G6 (offline xlsx parse, two-nearest-sample semantics, FileNotFound,
golden parity with legacy ``methods/site_LST.py::Hi``).

The reader is exercised fully offline via ``data_dir`` pointing at the
committed fixture; no network is involved (HiWATER data is local-only).

QC behaviour change (IMPORTANT):
    Legacy ``Hi`` averaged the two nearest samples with NO std>1 filter. The
    new pipeline runs the uniform ``decide_qc`` (std>1 → ``StdError``) on
    HiWATER's 2 samples too. Consequence: windows where the two nearest
    samples differ by >~1.4K now return ``StdError`` where legacy returned
    an average. The golden case below picks an overpass whose two nearest
    samples are within ~0.03K (std=0.0231 ≪ 1), so the math is validated
    and the QC flag is ``OK`` — matching legacy within the σ-unification
    tolerance (0.0085 K ≪ 0.01 K).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lstnet.ground_lst import SIGMA, lst_from_radiance
from lstnet.io.base import NetworkReader, RadiationWindow
from lstnet.io.hiwater import HiwaterReader, QC_FILE_NOT_FOUND, QC_OK
from lstnet.models import Site
from lstnet.qc import decide_qc

FIXTURES = Path(__file__).parent / "fixtures"

# 阿柔超级站 (arz), overpass 2018-07-01 02:43 UTC → +8 h → 10:43 Beijing.
# The fixture is a 30-row slice of the production file spanning 08:10..13:00
# Beijing (10-min interval); the two nearest samples to 10:43 are 10:40 and
# 10:50 (rows at offsets 15 and 16 in the slice).
_ARZ = Site(name="arz", network="HiWATER", lon=100.4643, lat=38.0473)
_OVERPASS_UTC = datetime(2018, 7, 1, 2, 43)
_WINDOW = 0  # ignored by HiwaterReader — two-nearest-sample semantics

# Legacy captured value (methods/site_LST.py::Hi('arz','201807010243',0.95))
# via py3-compatible re-run with retrieval math UNMODIFIED; legacy σ=5.6697e-8.
# Raw (un-.2f) mean = 284.831782 K; .2f formatted = 284.83 K.
_GOLDEN_LEGACY_LST_RAW = 284.831782

# New-pipeline expected value with unified SIGMA=5.670374e-8.
# Hand-computed from the two fixture rows:
#   row@10:40: ULR=372.6810, DLR=361.6454  →  T_a = 284.839665 K
#   row@10:50: ULR=372.4718, DLR=360.7170  →  T_b = 284.806970 K
#   avg = 284.823318 K, std(ddof=1) = 0.0231 → QC=OK.
_GOLDEN_NEW_LST_RAW = 284.823318

# Expected raw radiance samples at the golden overpass (from fixture rows).
_EXPECTED_L_UP_A = 372.6810
_EXPECTED_L_DOWN_A = 361.6454
_EXPECTED_L_UP_B = 372.4718
_EXPECTED_L_DOWN_B = 360.7170


def _reader() -> HiwaterReader:
    # ``data_dir`` points at the nested fixture layout (mirrors production
    # ``data/HiWATER/{year}/...``); the trimmed workbook lives at
    # ``tests/fixtures/hiwater/2018/2018年...阿柔超级站AWS.xlsx``.
    return HiwaterReader(data_dir=FIXTURES / "hiwater")


def test_satisfies_network_reader_protocol():
    assert isinstance(HiwaterReader(data_dir=FIXTURES / "hiwater"), NetworkReader)


def test_known_overpass_returns_ok_two_nearest_samples():
    """Two samples straddling 10:43 Beijing: 10:40 and 10:50."""
    win = _reader().read_radiation(_ARZ, _OVERPASS_UTC, _WINDOW)
    assert isinstance(win, RadiationWindow)
    assert win.status == QC_OK
    assert len(win.samples) == 2
    # The two sample times, in Beijing wall-clock (naive, +8h applied inside
    # the reader; reported back as UTC overpass-relative).
    times = sorted(s.time for s in win.samples)
    # Reader stores sample.time as the station row time MINUS 8h (back to UTC),
    # so 10:40 Beijing → 02:40 UTC, 10:50 Beijing → 02:50 UTC.
    assert times[0] == datetime(2018, 7, 1, 2, 40)
    assert times[1] == datetime(2018, 7, 1, 2, 50)


def test_samples_carry_radiance_values_from_fixture_rows():
    """l_up = ULR_Cor, l_down = DLR_Cor (HiWATER is radiance-based → direct)."""
    win = _reader().read_radiation(_ARZ, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_OK
    by_time = {s.time: s for s in win.samples}
    a = by_time[datetime(2018, 7, 1, 2, 40)]
    b = by_time[datetime(2018, 7, 1, 2, 50)]
    assert a.l_up == pytest.approx(_EXPECTED_L_UP_A, abs=1e-6)
    assert a.l_down == pytest.approx(_EXPECTED_L_DOWN_A, abs=1e-6)
    assert b.l_up == pytest.approx(_EXPECTED_L_UP_B, abs=1e-6)
    assert b.l_down == pytest.approx(_EXPECTED_L_DOWN_B, abs=1e-6)


def test_missing_file_returns_file_not_found_status():
    """No fixture for year 2017 → FileNotFound."""
    nowhere = Site(name="arz", network="HiWATER", lon=0.0, lat=0.0)
    win = _reader().read_radiation(nowhere, datetime(2017, 1, 1, 0, 0), _WINDOW)
    assert win.status == QC_FILE_NOT_FOUND
    assert win.samples == []


def test_unknown_site_name_returns_file_not_found():
    """Site not in the site→中文名 map → filename mismatch → FileNotFound."""
    unknown = Site(name="zzz", network="HiWATER", lon=0.0, lat=0.0)
    win = _reader().read_radiation(unknown, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_FILE_NOT_FOUND


def test_golden_sample_matches_legacy_hi():
    """Reproduce legacy Hi('arz','201807010243',0.95) == 284.83 K (raw 284.8318).

    The new pipeline (HiwaterReader → lst_from_radiance → decide_qc) must
    match the legacy raw mean within 0.01 K. σ unification (5.670374e-8 vs
    legacy 5.6697e-8) shifts the result by 0.0085 K — within the golden
    tolerance. std(ddof=1)=0.0231 ≪ 1 → QC=OK (legacy averaged unconditionally;
    the new pipeline applies decide_qc but the std gate passes here).
    """
    win = _reader().read_radiation(_ARZ, _OVERPASS_UTC, _WINDOW)
    assert win.status == QC_OK

    samples_k = [lst_from_radiance(s.l_up, s.l_down, 0.95) for s in win.samples]
    avg_k, qc = decide_qc(samples_k)
    assert qc == "OK"
    assert avg_k is not None
    # Matches the new-σ hand-computed mean essentially exactly.
    assert avg_k == pytest.approx(_GOLDEN_NEW_LST_RAW, abs=1e-6)
    # Matches legacy raw mean within σ-unification tolerance (0.0085 K < 0.01 K).
    assert avg_k == pytest.approx(_GOLDEN_LEGACY_LST_RAW, abs=0.01)


def test_qc_improvement_std_error_when_two_samples_diverge():
    """Documented QC behaviour change: std>1 → StdError (legacy averaged).

    HiWATER has only 2 nearest samples; ``decide_qc`` uses ddof=1 std.
    std > 1 ⟺ |T_a − T_b| > sqrt(2) ≈ 1.414 K. Construct two synthetic
    samples 2 K apart in LST space → StdError. Legacy ``Hi`` would have
    returned their average; the new pipeline rejects them uniformly with
    SURFRAD/PKU.
    """
    # 300 K and 302 K → mean 301, std(ddof=1)=1.414 > 1 → StdError.
    samples_k = [300.0, 302.0]
    avg_k, qc = decide_qc(samples_k)
    assert qc == "StdError"
    assert avg_k is None
