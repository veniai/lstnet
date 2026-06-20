"""HiWATER reader (Task 10).

Ports ``methods/site_LST.py::Hi`` (+ ``Hi_retrieval`` / ``Hi_nearest``) to
the :class:`NetworkReader` protocol.

HiWATER AWS stations log **up/down longwave radiation** (``ULR_Cor`` /
``DLR_Cor`` columns, W/m^2) at 10-minute intervals in ``.xlsx`` workbooks
named ``{year}年黑河流域地表过程综合观测网{中文名}AWS.xlsx``. The retrieval
is radiance-based: each row maps directly to a :class:`RadiationSample`
(``l_up = ULR_Cor``, ``l_down = DLR_Cor``) and feeds
:func:`lstnet.ground_lst.lst_from_radiance` — no equivalent-radiance trick
(unlike PKU's paired brightness temperatures).

Two-nearest-sample semantics
----------------------------
Legacy ``Hi`` binary-searched the workbook for the two rows straddling the
overpass time (``a``, ``b``) and averaged their LSTs. This reader returns
BOTH rows as :class:`RadiationSample`\\ s; the average + std gate happen
downstream in :func:`lstnet.qc.decide_qc`. When the overpass lands exactly
on a row time, legacy returned ``(mid, mid)`` — the same row twice — and
this reader preserves that (two identical samples → std=0 → OK).

Time handling
-------------
Station data is logged in Beijing time (UTC+8). The overpass time passed
in is UTC; ``+8 h`` is applied before searching (mirrors legacy).
``sample.time`` is reported back in UTC (station row time − 8 h) for
consistency with the overpass-centered window semantics used by
SURFRAD/PKU.

σ unification
-------------
Legacy ``Hi_retrieval`` used σ=5.6697e-8; the new pipeline uses the
unified :data:`lstnet.ground_lst.SIGMA` (5.670374e-8). The delta on LST
is ~0.0085 K for typical summer scenes — within the 0.01 K golden
tolerance.

QC behaviour change (IMPORTANT)
-------------------------------
Legacy ``Hi`` averaged the two nearest samples with NO std>1 filter. The
new pipeline runs the uniform :func:`lstnet.qc.decide_qc` (std>1 →
``StdError``) on HiWATER's 2 samples too. Consequence: windows where the
two nearest samples differ by >~1.4 K (std with ddof=1 > 1) now return
``StdError`` where legacy returned an average. This is an intentional,
more-consistent QC improvement — HiWATER now gets the same QC treatment
as SURFRAD and PKU. Golden cases pick overpasses whose two nearest
samples are within ~1 K so the math is validated (std < 1 → matches
legacy average within σ tolerance).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lstnet import config
from lstnet.io.base import RadiationSample, RadiationWindow
from lstnet.models import Site

# Legacy status strings (methods/site_LST.py returns these on failure).
QC_OK = "OK"
QC_FILE_NOT_FOUND = "FileNotFound"

# site name → 中文名 (legacy ``site_corrd`` in methods/site_LST.py::Hi).
# HiWATER = Heihe River Basin (黑河流域) observation network.
_SITE_CN = {
    "arz": "阿柔超级站",
    "dmz": "大满超级站",
    "dsl": "大沙龙站",
    "hhz": "黑河遥感站",
    "hzz": "花寨子站",
    "hmz": "荒漠站",
    "hjl": "混合林站",
    "jyl": "景阳岭站",
    "sdq": "四道桥超级站",
    "ykz": "垭口站",
    "zyz": "张掖湿地站",
}


def _nearest(arr: np.ndarray, target: datetime.datetime) -> tuple[int, int]:
    """Binary search for the two sample indices straddling ``target``.

    Matches legacy ``Hi_nearest`` exactly: on a sorted datetime array the
    recursive bisection returns ``(a, b)`` with ``a < b`` straddling the
    target, or ``(mid, mid)`` when the target lands on a row. Validated
    against the literal recursive port on a 200-row slice.

    ``arr`` is a ``datetime64[us]`` numpy array (naive wall-clock — the
    station logs Beijing time and legacy compares naive-to-naive).
    """
    # Strip tzinfo: np.datetime64 doesn't support aware datetimes (UserWarning).
    # The station data is naive Beijing local time; target was shifted +8h above.
    target_naive = target.replace(tzinfo=None) if target.tzinfo else target
    target_np = np.datetime64(target_naive).astype("datetime64[us]")
    arr_us = arr.astype("datetime64[us]")
    i = int(np.searchsorted(arr_us, target_np))
    if i < len(arr_us) and arr_us[i] == target_np:
        return i, i
    a = max(i - 1, 0)
    b = min(i, len(arr_us) - 1)
    if a == b:
        b = min(a + 1, len(arr_us) - 1)
    return a, b


class HiwaterReader:
    """Reads HiWATER ``.xlsx`` workbooks into :class:`RadiationWindow`.

    Files are resolved as ``data_dir / {year} / {year}年...{中文名}AWS.xlsx``
    (production layout); a flat fallback ``data_dir / {filename}``` is also
    checked so unit tests can point ``data_dir`` at a single trimmed
    fixture workbook. No network access is performed (HiWATER data is
    local-only).
    """

    network = "HiWATER"

    def __init__(self, data_dir: Path | str | None = None):
        # Default anchors at the CWD-independent repo root (G9); legacy
        # used a relative ``./data/HiWATER/`` which broke outside the repo.
        self.data_dir = (
            Path(data_dir) if data_dir is not None else config.project_root() / "data" / "HiWATER"
        )

    def read_radiation(
        self, site: Site, overpass_time: datetime.datetime, window_minutes: int
    ) -> RadiationWindow:
        """Return the two nearest radiance samples around ``overpass_time``.

        ``overpass_time`` is interpreted as UTC; the station logs in Beijing
        time (UTC+8), so +8 h is applied before searching the workbook
        (mirrors legacy). ``window_minutes`` is accepted for protocol parity
        but ignored: HiWATER windows are fixed at the two nearest samples.
        """
        tiftime = overpass_time + datetime.timedelta(hours=8)
        filepath = self._resolve(site, tiftime)
        if filepath is None:
            return RadiationWindow(
                site=site,
                overpass_time=overpass_time,
                samples=[],
                status=QC_FILE_NOT_FOUND,
            )

        ws = pd.read_excel(filepath)
        times = ws["TIMESTAMP"]
        arr = times.values  # datetime64[us], naive (Beijing wall-clock)
        ulr = ws["ULR_Cor"].to_numpy(dtype=float)
        dlr = ws["DLR_Cor"].to_numpy(dtype=float)

        a, b = _nearest(arr, tiftime)
        samples = [
            self._sample(times.iloc[a], ulr[a], dlr[a]),
            self._sample(times.iloc[b], ulr[b], dlr[b]),
        ]
        return RadiationWindow(
            site=site,
            overpass_time=overpass_time,
            samples=samples,
            status=QC_OK,
        )

    @staticmethod
    def _sample(
        station_time: pd.Timestamp, l_up: float, l_down: float
    ) -> RadiationSample:
        """Build a sample, reporting ``time`` back in UTC (station time − 8 h)."""
        utc_time = station_time.to_pydatetime() - datetime.timedelta(hours=8)
        return RadiationSample(time=utc_time, l_up=float(l_up), l_down=float(l_down))

    def _resolve(self, site: Site, tiftime: datetime.datetime) -> Path | None:
        """Locate the workbook; None if missing.

        Tries the year-prefixed production layout first, then a flat
        ``data_dir / filename`` (for trimmed test fixtures).
        """
        cn = _SITE_CN.get(site.name)
        if cn is None:
            return None
        year = f"{tiftime.year:04d}"
        filename = f"{year}年黑河流域地表过程综合观测网{cn}AWS.xlsx"
        nested = self.data_dir / year / filename
        if nested.exists():
            return nested
        flat = self.data_dir / filename
        if flat.exists():
            return flat
        return None
