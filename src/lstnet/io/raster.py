"""GeoTIFF raster helpers — extract a pixel value by lon/lat.

Used to ingest retrieved-LST rasters (the user's own algorithm output, or a
satellite LST product GeoTIFF): extract each station pixel into a table, then
feed :class:`~lstnet.validation.TableRetrievedLST` / :func:`~lstnet.validation.validate`.
This modernizes the legacy ``methods/Modis_emiss.py::findpointmodis`` pattern.

Uses ``rasterio`` (its pip wheel bundles GDAL — no system GDAL install; the
MODIS path stays on pyhdf with no GDAL at all).
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.windows import Window


def pixel_at_lonlat(path: Path | str, lon: float, lat: float, band: int = 1) -> float:
    """Return the raster value of the pixel containing ``(lon, lat)``.

    Assumes the raster is georeferenced in a lon/lat CRS (e.g. EPSG:4326).
    Raises :class:`ValueError` if the point falls outside the raster bounds.
    """
    with rasterio.open(path) as ds:
        col_f, row_f = ~ds.transform * (lon, lat)
        col = int(col_f)  # pixel containing the point (floor for non-negative)
        row = int(row_f)
        if not (0 <= col < ds.width and 0 <= row < ds.height):
            raise ValueError(
                f"(lon, lat)=({lon}, {lat}) -> pixel ({col}, {row}) is outside "
                f"raster bounds {ds.width}x{ds.height}"
            )
        window = Window(col_off=col, row_off=row, width=1, height=1)
        value = ds.read(band, window=window)  # shape (1, 1) for a 1x1 window
        return float(value[0, 0])
