"""Tests for the GeoTIFF pixel-extraction helper (lstnet.io.raster)."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lstnet.io.raster import pixel_at_lonlat


def _write_grid(path, data, west=0.0, north=3.0, pixel=1.0):
    """Write a lon/lat GeoTIFF (upper-left at (west, north), square pixels)."""
    arr = np.asarray(data, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs="EPSG:4326",
        transform=from_origin(west, north, pixel, pixel),
    ) as ds:
        ds.write(arr, 1)


def test_pixel_at_lonlat_returns_containing_pixel(tmp_path):
    # 3x3 grid covering lon[0,3] x lat[0,3], 1 deg pixels, UL at (0, 3):
    path = tmp_path / "r.tif"
    _write_grid(path, [[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    # (0.5, 2.5) -> pixel (col0,row0) = 10 ; (1.5, 1.5) -> (col1,row1) = 14
    assert pixel_at_lonlat(path, 0.5, 2.5) == 10.0
    assert pixel_at_lonlat(path, 1.5, 1.5) == 14.0
    assert pixel_at_lonlat(path, 2.4, 0.4) == 18.0  # (col2,row2)


def test_pixel_at_lonlat_outside_bounds_raises(tmp_path):
    path = tmp_path / "r.tif"
    _write_grid(path, [[10, 11], [13, 14]])
    with pytest.raises(ValueError):
        pixel_at_lonlat(path, 10.0, 10.0)
