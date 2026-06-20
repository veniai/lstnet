"""SURFRAD reader (Task 8).

Ports ``methods/site_LST.py::SURFRADlst`` to the ``NetworkReader`` protocol,
modernized: HTTPS download instead of FTP, timestamp-driven row selection
(robust to trimmed cache files), and decoupled I/O so the reader is fully
testable offline via ``data_dir``.

Column layout (whitespace-split rows, after the 2-line header):
    [16] = downwelling longwave radiation (W/m^2)  — ``l_down``
    [22] = upwelling longwave radiation   (W/m^2)  — ``l_up``
Row time fields: [0]=year [1]=doy [2]=month [3]=day [4]=hour [5]=minute.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typing import Callable

import requests

from lstnet import config
from lstnet.io.base import RadiationSample, RadiationWindow
from lstnet.models import Site
from lstnet.qc import QC_FILE_NOT_FOUND, QC_NO_DATA, QC_OK, QC_OUT_OF_DATE

#: Fetcher signature: ``(url, dest_path) -> None``; raises on failure.
Fetcher = Callable[[str, Path], None]

SURFRAD_BASE_URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad/"

# SURFRAD began 1995-01-01; sampling interval dropped from 3 min to 1 min at
# the 2009-01-01 cutover (mirrors methods/site_LST.py).
_INITIAL_DATE = datetime(1995, 1, 1, tzinfo=timezone.utc)
_UPDATED_DATE = datetime(2009, 1, 1, tzinfo=timezone.utc)

_L_DOWN_COL = 16
_L_UP_COL = 22
_HEADER_LINES = 2


class SurfradReader:
    """Reads SURFRAD ``.dat`` files into :class:`RadiationWindow` objects.

    Files are resolved as ``data_dir / site / YYYY / {site}{julian}.dat`` if
    missing locally; on a cache miss the file is fetched from
    :data:`SURFRAD_BASE_URL` via HTTPS and written under ``data_dir``.
    Pointing ``data_dir`` at a fixtures directory makes the reader fully
    offline-testable (the network path is never exercised in unit tests).
    """

    network = "SURFRAD"

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        fetcher: Fetcher | None = None,
    ):
        self.data_dir = (
            Path(data_dir) if data_dir is not None else config.project_root() / "data" / "SURFRAD"
        )
        self.base_url = SURFRAD_BASE_URL
        # Inject ``fetcher`` (e.g. a stub) to test cache-miss paths offline; the
        # default performs a real HTTPS GET via ``requests``.
        self._fetcher = fetcher or self._https_fetcher

    def read_radiation(
        self, site: Site, overpass_time: datetime, window_minutes: int
    ) -> RadiationWindow:
        if overpass_time.tzinfo is None:
            overpass_time = overpass_time.replace(tzinfo=timezone.utc)

        if overpass_time < _INITIAL_DATE:
            return RadiationWindow(
                site=site, overpass_time=overpass_time, samples=[], status=QC_OUT_OF_DATE
            )

        filepath = self._resolve(site, overpass_time)
        if filepath is None:
            return RadiationWindow(
                site=site,
                overpass_time=overpass_time,
                samples=[],
                status=QC_FILE_NOT_FOUND,
            )

        step = 3 if overpass_time < _UPDATED_DATE else 1
        rows = self._parse_rows(filepath)
        samples = self._select_window(rows, overpass_time, window_minutes, step)
        status = QC_OK if samples else QC_NO_DATA
        return RadiationWindow(
            site=site, overpass_time=overpass_time, samples=samples, status=status
        )

    def _resolve(self, site: Site, overpass_time: datetime) -> Path | None:
        """Locate the ``.dat`` file, fetching it over HTTPS if absent."""
        local = self._local_path(site, overpass_time)
        if local.exists():
            return local
        url = self._url(site, overpass_time)
        try:
            self._fetcher(url, local)
        except Exception:
            return None
        return local if local.exists() else None

    def _local_path(self, site: Site, overpass_time: datetime) -> Path:
        julian = overpass_time.year * 1000 + overpass_time.timetuple().tm_yday
        filename = f"{site.name}{str(julian)[2:]}.dat"
        return self.data_dir / site.name / f"{overpass_time.year:04d}" / filename

    def _url(self, site: Site, overpass_time: datetime) -> str:
        julian = overpass_time.year * 1000 + overpass_time.timetuple().tm_yday
        filename = f"{site.name}{str(julian)[2:]}.dat"
        return f"{self.base_url}{site.name}/{overpass_time.year:04d}/{filename}"

    @staticmethod
    def _https_fetcher(url: str, dest: Path) -> None:
        """Default HTTPS downloader (G11: never FTP)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    @staticmethod
    def _parse_rows(filepath: Path) -> list[dict]:
        """Return one dict per data row with time + l_down + l_up.

        Uses ``float`` (not ``eval``) for the radiation columns. Rows whose
        column count is too short or whose time fields are non-numeric are
        skipped, so partial/corrupt cache files degrade gracefully.
        """
        rows: list[dict] = []
        with filepath.open("r") as f:
            lines = f.readlines()
        for line in lines[_HEADER_LINES:]:
            parts = line.split()
            if len(parts) <= _L_UP_COL:
                continue
            try:
                year = int(parts[0])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                l_down = float(parts[_L_DOWN_COL])
                l_up = float(parts[_L_UP_COL])
            except (ValueError, IndexError):
                continue
            try:
                t = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                continue
            rows.append({"time": t, "l_down": l_down, "l_up": l_up})
        return rows

    @staticmethod
    def _select_window(
        rows: list[dict],
        overpass_time: datetime,
        window_minutes: int,
        step: int,
    ) -> list[RadiationSample]:
        """Pick samples within ±``window_minutes`` of the overpass.

        When ``window_minutes`` is 0 (caller delegates radius to the native
        step), fall back to the legacy row-radius of (5 - step) minutes —
        i.e. ±4 min for step=1 (9 samples) and ±2 min for step=3 (the legacy
        ``range(row-5+step, row+6-step)`` envelope).
        """
        radius = window_minutes if window_minutes > 0 else (5 - step)
        out: list[RadiationSample] = []
        for r in rows:
            delta_min = abs((r["time"] - overpass_time).total_seconds()) / 60.0
            if delta_min <= radius + 1e-9:
                out.append(
                    RadiationSample(
                        time=r["time"], l_up=r["l_up"], l_down=r["l_down"]
                    )
                )
        return out
