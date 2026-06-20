"""Tests for the validation engine: value objects, ingest, pairing, stats."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lstnet.models import (
    GroundLST,
    RetrievedLST,
    Site,
    ValidationPair,
    ValidationResult,
    ValidationStats,
)

_SITE = Site(name="psu", network="SURFRAD", lon=-77.93, lat=40.72)
_TIME = datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)


# --- T1: value objects -------------------------------------------------------

def test_retrieved_lst_default_source():
    r = RetrievedLST(overpass_time=_TIME, site=_SITE, lst_k=305.0)
    assert r.lst_k == 305.0
    assert r.source == "own-algorithm"


def test_validation_pair_diff_is_retrieved_minus_ground():
    g = GroundLST(_TIME, _SITE, 300.0, 0.97, "Day", "OK")
    r = RetrievedLST(_TIME, _SITE, 302.0, "algo")
    pair = ValidationPair(ground=g, retrieved=r, diff=r.lst_k - g.lst_k)
    assert pair.diff == 2.0


def test_validation_stats_and_result_construct():
    stats = ValidationStats(n=10, bias=0.5, rmse=1.2, r=0.9, slope=1.0, intercept=0.0)
    res = ValidationResult(pairs=[], unmatched_ground=[], unmatched_retrieved=[], stats=stats)
    assert res.stats.n == 10
    assert res.pairs == []


# --- T3: table ingest --------------------------------------------------------

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retrieved.csv"


def test_table_retrieved_loads_csv_12digit_time():
    from lstnet.validation import TableRetrievedLST

    tbl = TableRetrievedLST(_FIXTURE)
    items = tbl.items
    assert len(items) == 3
    assert items[0].site.name == "psu"
    assert items[0].lst_k == pytest.approx(305.0)
    assert items[0].source == "my-algo"
    assert items[0].overpass_time == datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)


def test_table_retrieved_parses_iso_time(tmp_path):
    from lstnet.validation import TableRetrievedLST

    csv = tmp_path / "iso.csv"
    csv.write_text("site,overpass_time_utc,lst_k\npsu,2020-07-14T14:13:00Z,305.0\n", encoding="utf-8")
    items = TableRetrievedLST(csv).items
    assert items[0].overpass_time == datetime(2020, 7, 14, 14, 13, tzinfo=timezone.utc)


def test_table_retrieved_missing_file_raises():
    from lstnet.validation import TableRetrievedLST

    with pytest.raises(FileNotFoundError):
        TableRetrievedLST(_FIXTURE.parent / "does-not-exist.csv")


# --- T4: pairing + engine ----------------------------------------------------

import math as _math
from datetime import timedelta as _timedelta

_DRA = Site(name="dra", network="SURFRAD", lon=-116.01947, lat=36.62373)
_TIME2 = datetime(2020, 7, 14, 14, 30, tzinfo=timezone.utc)


def _ground(site, t, k):
    return GroundLST(t, site, k, 0.97, "Day", "OK")


def _retr(site, t, k):
    return RetrievedLST(t, site, k, "algo")


def test_validate_pairs_by_site_and_time():
    from lstnet.validation import validate

    res = validate(
        [_ground(_SITE, _TIME, 300.0), _ground(_DRA, _TIME2, 295.0)],
        [_retr(_SITE, _TIME, 302.0), _retr(_DRA, _TIME2, 296.0)],
        time_tolerance_minutes=10,
    )
    assert len(res.pairs) == 2
    assert res.pairs[0].diff == pytest.approx(2.0)
    assert res.unmatched_ground == []
    assert res.unmatched_retrieved == []
    assert res.stats.bias == pytest.approx(1.5)


def test_validate_reports_unmatched_both_directions():
    from lstnet.validation import validate

    res = validate([_ground(_SITE, _TIME, 300.0)], [_retr(_DRA, _TIME2, 296.0)])
    assert len(res.pairs) == 0
    assert len(res.unmatched_ground) == 1
    assert len(res.unmatched_retrieved) == 1


def test_validate_tolerance_window_pairs_and_rejects():
    from lstnet.validation import validate

    close = _TIME + _timedelta(minutes=5)
    res = validate([_ground(_SITE, _TIME, 300.0)], [_retr(_SITE, close, 305.0)], time_tolerance_minutes=10)
    assert len(res.pairs) == 1
    far = _TIME + _timedelta(minutes=20)
    res2 = validate([_ground(_SITE, _TIME, 300.0)], [_retr(_SITE, far, 305.0)], time_tolerance_minutes=10)
    assert len(res2.pairs) == 0
    assert len(res2.unmatched_ground) == 1


def test_validate_skips_nan_ground():
    from lstnet.validation import validate

    bad = GroundLST(_TIME, _SITE, _math.nan, _math.nan, "Day", "StdError")
    good = _ground(_SITE, _TIME, 300.0)
    res = validate([bad, good], [_retr(_SITE, _TIME, 302.0)])
    assert len(res.pairs) == 1
