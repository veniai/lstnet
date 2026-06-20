"""LSTNet MCP server — expose LST computation / validation as tools for AI agents.

Run: ``lstnet-mcp`` or ``python -m lstnet.mcp_server``. Connect from any MCP
client (Claude Desktop, an agent, etc.) to compute ground-truth LST and run
validation through the :mod:`lstnet` library.

Tools:
  - ``list_sites`` — the 25 validation sites.
  - ``compute_lst`` — ground-truth LST at one site + overpass time.
  - ``validate_csv`` — validate a ground-LST CSV against a retrieved-LST CSV.
"""
from __future__ import annotations

import csv as _csv
from datetime import datetime, timezone

from fastmcp import FastMCP

from lstnet import FixedEmissivity, compute_ground_lst, validate
from lstnet.io.hiwater import HiwaterReader
from lstnet.io.pku import PkuReader
from lstnet.io.surfrad import SurfradReader
from lstnet.models import GroundLST
from lstnet.sites import SITES, get_site
from lstnet.validation import TableRetrievedLST

mcp = FastMCP("lstnet")
_READERS = {"SURFRAD": SurfradReader, "PKULSTNet": PkuReader, "HiWATER": HiwaterReader}


def _clean(x):
    """JSON-safe float (NaN -> None)."""
    return None if (isinstance(x, float) and x != x) else x


def _parse_emissivity(spec: str):
    """Parse an emissivity spec: ``fixed:VALUE`` | ``modis_daily`` | ``aster_ged``."""
    spec = (spec or "").strip()
    if spec.startswith("fixed:"):
        return FixedEmissivity(float(spec.split(":", 1)[1]))
    if spec == "modis_daily":
        from lstnet import ModisDailyEmissivity
        return ModisDailyEmissivity()
    if spec == "aster_ged":
        from lstnet import AsterGEDEmissivity
        return AsterGEDEmissivity()
    return FixedEmissivity(0.95)


@mcp.tool
def list_sites() -> list[dict]:
    """List all 25 validation sites (name, network, lon, lat)."""
    return [
        {"name": s.name, "network": s.network, "lon": s.lon, "lat": s.lat}
        for s in SITES.values()
    ]


@mcp.tool
def compute_lst(site: str, overpass_time: str, emissivity: str = "fixed:0.95") -> dict:
    """Compute ground-truth LST at one site for one overpass moment.

    Args:
        site: site name, e.g. ``"psu"`` (see ``list_sites``).
        overpass_time: ``YYYYMMDDHHMM`` in UTC, e.g. ``"201102121430"``.
        emissivity: ``"fixed:VALUE"`` (offline), ``"modis_daily"`` (needs Earthdata
            creds), or ``"aster_ged"`` (needs creds). Default ``"fixed:0.95"``.

    Returns ``{lst_k, emissivity, qc_flag, day_or_night}``.
    """
    s = get_site(site)
    t = datetime.strptime(overpass_time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    em = _parse_emissivity(emissivity)
    reader = _READERS[s.network]()
    g = compute_ground_lst(s, t, em, reader)
    return {
        "lst_k": _clean(g.lst_k),
        "emissivity": _clean(g.emissivity),
        "qc_flag": g.qc_flag,
        "day_or_night": g.day_or_night,
    }


@mcp.tool
def validate_csv(ground_csv: str, retrieved_csv: str) -> dict:
    """Validate a ground-LST CSV against a retrieved-LST CSV.

    ``ground_csv``: the GUI/library export (columns ``overpass_time, site,
    lst_k[, emissivity, qc]``). ``retrieved_csv``: columns
    ``site, overpass_time_utc, lst_k[, source]``.

    Returns ``{n, bias, rmse, r, unmatched_ground, unmatched_retrieved}``.
    """
    ground = _load_ground_csv(ground_csv)
    retrieved = TableRetrievedLST(retrieved_csv).items
    result = validate(ground, retrieved)
    s = result.stats
    return {
        "n": s.n,
        "bias": _clean(s.bias),
        "rmse": _clean(s.rmse),
        "r": _clean(s.r),
        "unmatched_ground": len(result.unmatched_ground),
        "unmatched_retrieved": len(result.unmatched_retrieved),
    }


def _load_ground_csv(path: str) -> list[GroundLST]:
    out: list[GroundLST] = []
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            site = get_site(row["site"])
            t = datetime.strptime(row["overpass_time"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            try:
                lst = float(row["lst_k"])
            except (ValueError, KeyError):
                lst = float("nan")
            try:
                em = float(row.get("emissivity", "nan"))
            except ValueError:
                em = float("nan")
            out.append(GroundLST(t, site, lst, em, "Unknown", row.get("qc", "OK")))
    return out


def main():
    mcp.run()


if __name__ == "__main__":
    main()
