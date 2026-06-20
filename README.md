# lstnet

**Pure-Python land surface temperature (LST) ground-validation library + GUI.**

`lstnet` computes **ground-truth LST** from in-situ longwave radiation at
SURFRAD / PKULSTNet / HiWATER validation sites and **validates** it against
retrieved LST (your own algorithm or a satellite product) — bias / RMSE / R +
scatter and time-series plots. Zero system GDAL; cross-platform
(Windows / macOS incl. Apple Silicon / Linux); MIT-licensed.

**Networks:** **SURFRAD** (7 US sites, served online by NOAA — the default,
works out of the box) is the primary network. **PKULSTNet** (7) and **HiWATER**
(11) readers are also included for users who hold those local datasets (they
are not publicly downloadable — point the reader at your `data/` directory).

## Install

```bash
pip install lstnet
```

That's it — core library, GUI (`lstnet-gui`), and MCP server (`lstnet-mcp`) are
all included.

For MODIS / ASTER GED emissivity sources, set NASA Earthdata credentials
(never hardcoded):

```bash
export EARTHDATA_USERNAME=you@example.com
export EARTHDATA_PASSWORD=********
```

Or enter them in the GUI: **Settings → Earthdata Login** (saved to
`~/.lstnet/earthdata.json`, chmod 600). Register at
<https://urs.earthdata.nasa.gov/users/new>.

`FixedEmissivity` needs no credentials.

## GUI usage

```bash
lstnet-gui        # launch the PySide6 desktop application
```

The GUI lets you:

1. **Select sites** (multi-select, with lon/lat shown) from SURFRAD / PKULSTNet
   / HiWATER.
2. **Enter overpass times** (12-digit `YYYYMMDDHHMM`, one per line, UTC).
3. **Pick an emissivity source** — ASTER GED (default, recommended for
   validation), MODIS daily, or a manual fixed value.
4. **Compute ground LST** — batch across all selected sites × times.
5. **Validate** — load a retrieved-LST CSV (`site, overpass_time_utc, lst_k`),
   the tool auto-computes ground truth for each row, pairs, and shows
   bias / RMSE / R + an embedded scatter plot (retrieved vs ground, 1:1 line).
6. **Export** the enriched CSV (retrieved LST + ground LST + diff + emissivity).

**Sample data** is in `samples/`:
- `retrieved_sample.csv` — 9-site multi-network demo (SURFRAD ×7 + HiWATER ×2).
- `validation_template.csv` — all 25 sites pre-filled (fill in time + your LST).

**Linux note:** the GUI needs `libxcb-cursor0`:
```bash
sudo apt install -y libxcb-cursor0 libegl1 libgl1
```

## Library usage (script / notebook)

Compute ground-truth LST for one site/time:

```python
from datetime import datetime, timezone
from lstnet import compute_ground_lst, FixedEmissivity
from lstnet.sites import get_site
from lstnet.io.surfrad import SurfradReader

site = get_site("psu")
t = datetime(2011, 2, 12, 14, 30, tzinfo=timezone.utc)
g = compute_ground_lst(site, t, FixedEmissivity(0.98), SurfradReader())
print(g.lst_k, g.qc_flag)   # e.g. 270.15 OK
```

Validate ground truth against your retrieved LST:

```python
from lstnet import validate, TableRetrievedLST
from lstnet.plotting import scatter_plot

ground = [compute_ground_lst(s, t, FixedEmissivity(0.98), SurfradReader())
          for s, t in your_site_times]
result = validate(ground, TableRetrievedLST("your_retrieval.csv"))
print(result.stats.bias, result.stats.rmse, result.stats.r)
scatter_plot(result)   # retrieved-vs-ground with 1:1 line
```

`your_retrieval.csv` columns: `site, overpass_time_utc, lst_k[, source]`
(`overpass_time_utc` = 12-digit `YYYYMMDDHHMM` or ISO 8601, UTC).

## MCP server (AI agent integration)

```bash
lstnet-mcp        # start the FastMCP server
```

Exposes three tools that an AI agent (Claude Desktop, etc.) can call:
`list_sites`, `compute_lst`, `validate_csv`.

## Features

- **Ground-truth LST** from SURFRAD (NOAA, HTTPS) / PKULSTNet / HiWATER
  station data, at satellite overpass times (Stefan–Boltzmann radiance inversion).
- **Emissivity sources**: `FixedEmissivity` (offline), `ModisDailyEmissivity`
  (MYD21A1D/A1N C6.1, Ogawa 2004 broadband), `AsterGEDEmissivity`
  (AG100 V003 climatological, Cheng & Liang 2014 broadband — recommended for
  sub-K validation).
- **Validation engine**: pair ground-truth + retrieved LST by `(site, overpass
  time)` within a tolerance; bias / RMSE / Pearson R / regression; scatter +
  time-series plots.
- **GDAL-free**: MODIS HDF4 via `pyhdf`, ASTER GED HDF5 via `h5py`, GeoTIFF via
  `rasterio` — all ship pip wheels, no system GDAL install.
- Quality control (configurable `strict`), day/night (astral), credential
  externalization, CWD-independent paths.
- **Cross-platform**: tested on Windows / macOS / Linux via GitHub Actions CI.

## Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✅ Full | GUI needs `libxcb-cursor0 libegl1 libgl1` (apt install) |
| macOS (Intel + Apple Silicon) | ✅ CI green | Native Qt backend; no extra libs |
| Windows | ✅ CI green | Native Qt backend; no extra libs |

All binary dependencies (pyhdf, h5py, rasterio, PySide6) ship pip wheels for
all three platforms — no compilation needed.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
