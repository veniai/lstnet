"""PKULSTNet reader (Task 9).

Ports ``methods/site_LST.py::PKULSTNet`` (+``_retrieval``/``_nearest``/``_datatest``)
to the :class:`NetworkReader` protocol.

PKU CR200Series loggers report *paired brightness temperatures* (TagTemp_C_Avg(1)
and TagTemp_C_Avg(2)), not longwave radiance. The legacy retrieval is::

    T = ((t1**4 - (1-emiss)*(t2**4)) / emiss) ** 0.25        # no sigma

To keep the uniform ``lst_from_radiance`` pipeline (and the
:class:`RadiationSample` / :class:`NetworkReader` interfaces that T7/T8 rely
on) we emit each sample as *equivalent* longwave radiance::

    l_up   = SIGMA * t1**4
    l_down = SIGMA * t2**4

so that ``lst_from_radiance(l_up, l_down, emiss)`` reduces (σ cancels) to
exactly the legacy paired formula. Algebra:

    ((σ·t1⁴ − (1−emiss)·σ·t2⁴) / (emiss·σ)) ** 0.25
  = ((t1⁴ − (1−emiss)·t2⁴) / emiss) ** 0.25           ✓ identical to legacy

The **hbc (承德站) t1/t2 swap** is applied BEFORE the radiance conversion: for
hbc, ``t1 = TagTemp_C_Avg(1)`` and ``t2 = TagTemp_C_Avg(2)``; for every other
PKU site the two are swapped (mirrors legacy ``_retrieval``).
"""
from __future__ import annotations

import datetime
import glob
from pathlib import Path

import numpy as np

from lstnet import config
from lstnet.ground_lst import SIGMA
from lstnet.io.base import RadiationSample, RadiationWindow
from lstnet.models import Site

# Legacy status strings (methods/site_LST.py returns these on failure).
QC_OK = "OK"
QC_DATA_ERROR = "DataError"
QC_FILE_NOT_FOUND = "FileNotFound"

# site name → 中文名 (legacy ``site_corrd``).
_SITE_CN = {
    "hbc": "承德站",
    "hnw": "海南站",
    "hnh": "鹤壁站",
    "xzl": "林芝站",
    "imb": "内蒙古站",
    "xjf": "新疆站",
    "cqb": "重庆站",
}

# Column indices into the comma-delimited row (after skip_header=4):
#   [0] TIMESTAMP, [1] RECORD, [2] TagTemp_C_Avg(1), [3] TagTemp_C_Avg(2), ...
_COL_T1_DEFAULT = 3   # non-hbc: t1 comes from TagTemp_C_Avg(2)
_COL_T2_DEFAULT = 2   # non-hbc: t2 comes from TagTemp_C_Avg(1)
_COL_T1_HBC = 2       # hbc: t1 comes from TagTemp_C_Avg(1)  (the swap)
_COL_T2_HBC = 3       # hbc: t2 comes from TagTemp_C_Avg(2)

# Per-interval branch parameters (ported verbatim from legacy lines 142-198):
#   (delta_threshold_min, slice_lo, slice_hi, min_valid_count)
# Legacy slices (Python half-open): 1-min → data[a-4:b+5], 2-min → data[a-2:b+3],
# 3-min → data[a-1:b+2]. We store slice_lo (a's offset) and slice_hi (b's
# offset); the actual slice is ``data[a-slice_lo : b+slice_hi]``.
# ``min_valid_count`` is the legacy count gate threshold (gate: ``len > N``).
_INTERVALS = {
    1: (5, 4, 5, 2),   # 1-min:  threshold 5 min, data[a-4:b+5], gate len > 2
    2: (6, 2, 3, 1),   # 2-min:  threshold 6 min, data[a-2:b+3], gate len > 1
    3: (9, 1, 2, 1),   # 3-min:  threshold 9 min, data[a-1:b+2], gate len > 1
}


def _parse_timestamp(raw: object) -> datetime.datetime:
    """Parse ``"YYYY-MM-DD HH:MM:SS"`` from a genfromtxt cell (bytes or str)."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return datetime.datetime.strptime(raw.strip('"'), "%Y-%m-%d %H:%M:%S")


def _nearest(
    data: np.ndarray, time: datetime.datetime, start: int, end: int
) -> tuple[int, int]:
    """Binary search for the nearest sample indices (legacy ``_nearest``)."""
    if end - start <= 1:
        return start, end
    mid = (start + end) // 2
    mid_time = data[mid][0]
    if mid_time > time:
        return _nearest(data, time, start, mid)
    elif mid_time < time:
        return _nearest(data, time, mid, end)
    else:
        return mid, mid


def _within_threshold(
    row_time: datetime.datetime, tiftime: datetime.datetime, threshold_min: int
) -> bool:
    """Legacy ``_datatest``: True iff |row_time - tiftime| <= threshold minutes."""
    return abs(row_time - tiftime) <= datetime.timedelta(minutes=threshold_min)


def brightness_to_radiance(row: np.ndarray, site_name: str) -> tuple[float, float]:
    """Convert a PKU row's brightness temperatures to equivalent longwave radiance.

    Applies the **hbc t1/t2 swap** before computing ``SIGMA * t**4`` so that
    ``lst_from_radiance`` reproduces the legacy paired-brightness formula.
    Returns ``(l_up, l_down)`` where ``l_up = σ·t1⁴`` and ``l_down = σ·t2⁴``.
    """
    if site_name == "hbc":
        t1 = float(row[_COL_T1_HBC]) + 273.15
        t2 = float(row[_COL_T2_HBC]) + 273.15
    else:
        t1 = float(row[_COL_T1_DEFAULT]) + 273.15
        t2 = float(row[_COL_T2_DEFAULT]) + 273.15
    return SIGMA * t1**4, SIGMA * t2**4


class PkuReader:
    """Reads PKULSTNet CR200Series ``.dat`` files into :class:`RadiationWindow`.

    Files are comma-delimited with a 4-line header. The reader is fully
    offline-testable via ``data_dir`` pointing at a fixtures directory; no
    network or remote fetch is performed (PKU data is local-only).
    """

    network = "PKULSTNet"

    def __init__(self, data_dir: Path | str | None = None):
        # Default anchors at the CWD-independent repo root (G9); legacy
        # used a relative ``./data/pku-sites/`` which broke outside the repo.
        self.data_dir = (
            Path(data_dir) if data_dir is not None else config.project_root() / "data" / "pku-sites"
        )

    def read_radiation(
        self, site: Site, overpass_time: datetime.datetime, window_minutes: int
    ) -> RadiationWindow:
        """Return equivalent-radiance samples around ``overpass_time``.

        ``overpass_time`` is interpreted as UTC; the station logs in Beijing
        time (UTC+8), so +8 h is applied before searching the file (mirrors
        legacy). ``window_minutes`` is accepted for protocol parity but
        ignored: PKU windows are fixed by the per-interval branch (1/2/3-min).
        """
        # PKU station data is Beijing local time (naive); strip tzinfo after +8h
        # so comparisons with the (naive) data timestamps don't raise.
        tiftime = (overpass_time + datetime.timedelta(hours=8)).replace(tzinfo=None)
        yearmonth = int(tiftime.strftime("%Y%m"))
        pattern = (
            self.data_dir
            / f"Total{_SITE_CN.get(site.name, site.name)}CR200Series_Data-{yearmonth}01*"
        )
        matches = glob.glob(str(pattern))
        if not matches:
            return RadiationWindow(
                site=site,
                overpass_time=overpass_time,
                samples=[],
                status=QC_FILE_NOT_FOUND,
            )

        data = np.genfromtxt(
            matches[0],
            delimiter=",",
            skip_header=4,
            converters={0: _parse_timestamp},
            dtype=None,
            encoding=None,
        )
        a, b = _nearest(data, tiftime, 0, len(data))

        for interval in _INTERVALS:
            params = _INTERVALS[interval]
            _, lo, hi, _ = params
            # Legacy accesses data[a-1] (step_below), data[b+1] (step_above),
            # and slices data[a-lo : b+hi]. Guard all of those indices.
            if a - lo < 0 or a - 1 < 0 or b + 1 >= len(data) or b + hi > len(data):
                continue
            step_below = data[a][0] - data[a - 1][0]
            step_above = data[b + 1][0] - data[b][0]
            target = datetime.timedelta(minutes=interval)
            if step_below == target and step_above == target:
                return self._collect_branch(
                    site, overpass_time, tiftime, data, a, b, interval, params
                )
        return RadiationWindow(
            site=site,
            overpass_time=overpass_time,
            samples=[],
            status=QC_DATA_ERROR,
        )

    def _collect_branch(
        self,
        site: Site,
        overpass_time: datetime.datetime,
        tiftime: datetime.datetime,
        data: np.ndarray,
        a: int,
        b: int,
        interval: int,
        params: tuple[int, int, int, int],
    ) -> RadiationWindow:
        """Run one of the three per-interval branches verbatim from legacy.

        Mirrors legacy lines 142-198 exactly: threshold check on every window
        row → build LST list → nan-filter → per-interval count gate. The
        reader returns the equivalent-radiance samples; LST computation uses
        ``lst_from_radiance`` (σ cancels → identical to legacy paired formula).
        """
        threshold_min, lo, hi, min_count = params
        window = data[a - lo : b + hi]

        # Legacy ``Effective.all()`` — every window row must be within the
        # branch's minute-threshold of the overpass, else DataError.
        effective = [_within_threshold(r[0], tiftime, threshold_min) for r in window]
        if not all(effective):
            return RadiationWindow(
                site=site,
                overpass_time=overpass_time,
                samples=[],
                status=QC_DATA_ERROR,
            )

        # Build samples (equivalent radiance). The legacy nan-filter
        # (``LSTlist[~isnan(LSTlist)]``) operates in LST space, but its NaNs
        # come entirely from NaN brightness temperatures in the source row —
        # so filtering on the input t1/t2 is equivalent and avoids pinning a
        # throwaway emissivity inside the reader.
        samples: list[RadiationSample] = []
        valid_count = 0
        for row in window:
            t1_idx = _COL_T1_HBC if site.name == "hbc" else _COL_T1_DEFAULT
            t2_idx = _COL_T2_HBC if site.name == "hbc" else _COL_T2_DEFAULT
            t1_raw = row[t1_idx]
            t2_raw = row[t2_idx]
            l_up, l_down = brightness_to_radiance(row, site.name)
            samples.append(
                RadiationSample(
                    time=row[0] - datetime.timedelta(hours=8),  # back to UTC
                    l_up=l_up,
                    l_down=l_down,
                )
            )
            if not (np.isnan(t1_raw) or np.isnan(t2_raw)):
                valid_count += 1

        # Legacy per-interval count gate (lines 151 / 169 / 187):
        #   1-min → ``len > 2``, 2/3-min → ``len > 1``.
        if valid_count <= min_count:
            return RadiationWindow(
                site=site,
                overpass_time=overpass_time,
                samples=[],
                status=QC_DATA_ERROR,
            )

        return RadiationWindow(
            site=site,
            overpass_time=overpass_time,
            samples=samples,
            status=QC_OK,
        )
