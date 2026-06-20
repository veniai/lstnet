"""Quality-control decisions for ground-LST sample windows.

``decide_qc`` emits only the sample-based flags (``OK`` / ``StdError`` / ``NoData``).
The remaining qc flags are set by other layers:
  - ``FileNotFound``  — network reader (missing data file)
  - ``TimeError``     — orchestration (unparseable overpass time)
  - ``OutOfDate``     — SURFRAD reader (overpass before 1995-01-01)
"""
from __future__ import annotations

import numpy as np

QC_OK = "OK"
QC_STD_ERROR = "StdError"
QC_NO_DATA = "NoData"

QC_FILE_NOT_FOUND = "FileNotFound"
QC_TIME_ERROR = "TimeError"
QC_OUT_OF_DATE = "OutOfDate"

# Legacy std>1K filter (methods/site_LST.py used np.std(ddof=1) > 1).
_STD_THRESHOLD = 1.0


def decide_qc(samples: list[float]) -> tuple[float | None, str]:
    """Return ``(average_lst_or_None, qc_flag)`` for a window of LST samples.

    Fewer than 2 samples → ``NoData`` (std with ddof=1 is undefined).
    Sample std > 1 K     → ``StdError``.
    Otherwise            → ``(mean, OK)``.
    """
    if len(samples) < 2:
        return None, QC_NO_DATA
    arr = np.asarray(samples, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std > _STD_THRESHOLD:
        return None, QC_STD_ERROR
    return float(np.average(arr)), QC_OK
