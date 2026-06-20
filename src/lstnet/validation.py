"""Validation engine: pair ground-truth LST with retrieved LST, compute stats.

Plan 1c scope: a table-based retrieved-LST ingest (``TableRetrievedLST``) and the
``validate`` engine that pairs by ``(site, overpass_time)`` within a tolerance.
A raster "extract station pixel" ingest is a follow-up once a real retrieval
raster format is supplied.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from lstnet.models import (
    GroundLST,
    RetrievedLST,
    ValidationPair,
    ValidationResult,
)
from lstnet.sites import get_site
from lstnet.stats import compute_stats


def _parse_overpass_time(value) -> datetime:
    """Parse ``YYYYMMDDHHMM`` (12-digit) or ISO 8601 into an aware UTC datetime."""
    s = str(value).strip()
    if len(s) == 12 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    ts = pd.to_datetime(s, utc=True)
    return ts.to_pydatetime().astimezone(timezone.utc)


class TableRetrievedLST:
    """Retrieved LST loaded from a CSV/Excel table.

    Columns: ``site``, ``overpass_time_utc`` (12-digit ``YYYYMMDDHHMM`` or ISO),
    ``lst_k``, and optional ``source``. ``site`` must be a name in the registry.
    """

    def __init__(self, path: Path | str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Retrieved-LST table not found: {path}")
        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
        self.items: list[RetrievedLST] = []
        for _, row in df.iterrows():
            site = get_site(str(row["site"]))
            overpass_time = _parse_overpass_time(row["overpass_time_utc"])
            self.items.append(
                RetrievedLST(
                    overpass_time=overpass_time,
                    site=site,
                    lst_k=float(row["lst_k"]),
                    source=str(row.get("source", "own-algorithm")),
                )
            )


def validate(
    ground: Iterable[GroundLST],
    retrieved: Iterable[RetrievedLST],
    time_tolerance_minutes: int = 10,
) -> ValidationResult:
    """Pair ground-truth LST with retrieved LST and compute validation stats.

    Each ground observation is paired with the nearest retrieved observation at
    the **same site** within ``time_tolerance_minutes`` of its overpass time
    (one-to-one; a retrieved value is used at most once). Ground observations
    whose ``lst_k`` is NaN (non-OK qc) are skipped. Unmatched ground and
    retrieved are reported separately — never dropped silently.
    """
    import math

    retrieved_list = list(retrieved)
    used: set[int] = set()
    pairs: list[ValidationPair] = []
    unmatched_ground: list[GroundLST] = []

    for g in ground:
        if math.isnan(g.lst_k):
            continue  # invalid ground truth (non-OK qc) — not pairable, not counted
        best_i = None
        best_dt = None
        for i, r in enumerate(retrieved_list):
            if i in used or r.site.name != g.site.name:
                continue
            dt = abs((r.overpass_time - g.overpass_time).total_seconds())
            if dt <= time_tolerance_minutes * 60 and (best_dt is None or dt < best_dt):
                best_i, best_dt = i, dt
        if best_i is None:
            unmatched_ground.append(g)
        else:
            used.add(best_i)
            r = retrieved_list[best_i]
            pairs.append(ValidationPair(ground=g, retrieved=r, diff=r.lst_k - g.lst_k))

    unmatched_retrieved = [r for i, r in enumerate(retrieved_list) if i not in used]
    return ValidationResult(
        pairs=pairs,
        unmatched_ground=unmatched_ground,
        unmatched_retrieved=unmatched_retrieved,
        stats=compute_stats(pairs),
    )
