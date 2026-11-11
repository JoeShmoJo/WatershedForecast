# Watershed Forecast

Builds basin precipitation maps, hourly/cumulative precipitation charts, and
stream gage charts for the basins in `Config.json`. GitHub Actions builds the
site and deploys a validated artifact to GitHub Pages.

## Run locally

Use Python 3.11. From the existing repository checkout:

```sh
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m unittest discover -s tests -v
python build_site.py --basin Canyon_Creek_WA --output .build/one-basin
python build_site.py --output .build/site
```

Open the generated `index.html` or serve the generated directory with a local
HTTP server. `python WatershedForecast.py --all` is a compatibility entry point
for the new builder. It also writes `.build/site`, rather than modifying tracked
files under `docs/`. No API keys, Chromium, system package installation, or
StreamStats requests are required. Every configured basin needs a cached
`Basins/<name>/basin.geojson`. Ring intersections are repaired in memory.

The default build offers **1 day, 2 days, and 3 days** on the existing page.
Selecting a duration updates every basin map, subbasin total, and precipitation
chart from the same model cycle. Gage charts continue showing the last 24 hours
of observations. Forecast start/end times are displayed in the viewer's local
timezone. The separate `portal_experiment/` prototype is not imported, built,
or deployed by this workflow.

Use `--days 1 2 3` to choose available day buttons, or `--hours` with 1–36, 48,
or 72 to build a single custom duration. `--workers` controls download concurrency,
capped at four. `--upscale` controls the plotting/aggregation grid subdivision
(default two). Output and cache paths are relative to the working directory.
Only parent names are accepted by `--basin`; their subbasins are included.

## Forecast data and calculations

NOAA retired the NOMADS OPeNDAP service used by the original downloader. The
replacement reads NBM CONUS core GRIB2 data from the public
`noaa-nbm-grib2-pds` AWS bucket, with NOMADS HTTPS as a discovery fallback.
No AWS credentials are required.

The builder selects a recent six-hour cycle, checking the first and final
requested forecast-hour indexes. It requires every requested hour to download
and decode successfully before publishing; a partially available cycle fails
the build. It never selects a cycle more than 18 hours old. HTTP failures,
missing indexes, and incorrect accumulation metadata produce explicit errors.

Only deterministic **surface APCP** messages are selected; probability fields
are excluded. Three-day builds preserve one-hour accumulations for the first
24 hours, followed by explicit, nonoverlapping six-hour accumulations through
72 hours. Later files also contain isolated one-hour amounts; selecting those
alone would leave gaps, so longer-range intervals are checked explicitly.
Day totals are nested prefixes of that same interval sequence and never double
count rainfall. The charts show each bar's actual interval duration and draw
it over that interval; six-hour totals are not labeled as hourly rates. Single
custom builds of up to 36 hours retain entirely hourly resolution.
The old six-hour boundary subtraction is gone. Native Lambert conformal coordinates and alternating row
scanning are decoded explicitly, then precipitation is linearly interpolated
onto a regional 0.025-degree latitude/longitude grid. Existing bilinear basin
sampling and cosine-latitude area weighting are reused. This is an interpolated
basin mean, not exact polygon/cell overlap integration; finer plotting does not
add forecast resolution. Liquid-equivalent millimeters are converted to inches
by dividing by 25.4.

All basins share one downloaded forecast. `.cache/nbm` retains indexed GRIB
messages for reuse, pruning matching cached messages older than two days before
the selected cycle. Static maps use Matplotlib directly; interactive charts use
Plotly. Kaleido, netCDF4, pydap, and rtree are no longer direct dependencies.

Each basin and duration gets PNG/SVG maps, interactive HTML charts, and
`precipitation.csv` with UTC interval-ending timestamps and `interval_hours`.
The first duration uses the original asset paths; additional durations use
`48h/` and `72h/` subdirectories. Entirely hourly outputs also include the
backward-compatible `hourly_precip.csv`. `status.json` records the actual model
initialization, forecast interval, source, and each gage's availability/latest
valid observation. The webpage marks forecasts older than 18 hours and
observations older than three hours as stale, including when updates stop.

The map card uses Leaflet over Esri's public World Topographic Map, with
permanent subbasin labels/totals, clickable basin summaries, zoom controls,
fit-to-basin, and a precipitation visibility toggle. Its overlay is rendered
from the **same accumulated NBM data** as the existing PNG and charts. It does
not use the separate WPC portal experiment or change the basin mean calculation.
`basin_map.py` contains this display-only renderer and `map_assets/` contains
its JavaScript/CSS and vendored Leaflet 1.9.4 with its license. Web Mercator
display pixels are masked by basin geometry; interpolation adds no model
resolution. Dry cells are transparent, and missing precipitation fails the build.
The main day controls switch maps and charts together. PNG/SVG exports remain
available, including when internet access to the background tiles is unavailable.
Map tiles require internet and include the provider's attribution; the forecast
overlay, geometry, controls, and calculations are generated static assets.

## Gage availability

Observations are fetched independently before forecast retrieval. Missing-data
sentinels are removed. An unavailable gage is labeled explicitly and does not
block otherwise valid precipitation forecasts. Observations and their status
remain in failed-build diagnostics even when forecasting fails.

Carson now uses NOAA NWPS gauge **WCNW1**, replacing the discontinued USGS flow
feed. NOAA's `kcfs` readings are converted to `cfs` for the chart. During live
validation on November 11, 2026, its latest reading was 529 cfs at 3:15 a.m.
Pacific, so it was correctly shown as stale rather than unavailable. Amboy,
Hatchery, Underwood, and EFL Heisson returned recent observations. Washougal
returned no flow series, and both WA Ecology TXT downloads returned HTTP 403
from their upstream application gateway. Their availability may change.

## Deploy to the existing GitHub Pages URL

1. Review and merge these changes into `main`.
2. In the repository's **Settings → Pages → Build and deployment**, change
   **Source** to **GitHub Actions**. The repository must permit Actions and Pages
   deployment. No personal access token is needed: the deploy job uses the
   built-in `GITHUB_TOKEN` and OIDC permissions.
3. Run **Actions → Build and publish plots → Run workflow** on `main`, or let
   the schedule run. The Pages address remains the repository's existing Pages
   URL unless its domain settings are changed.

Pull requests run offline regression tests. Main-branch pushes, scheduled runs,
and manual runs build live data; only `main` deploys. The workflow uses Ubuntu
24.04, Python 3.11, and hash-locked packages. It does not commit generated plots
or mutate the source branch. The old tracked `docs/` files remain as historical
data and are not included in the newly built Pages artifact.

Builds stage their output separately and validate every basin before replacing
a previous local artifact. A forecast, rendering, or validation error exits
nonzero, prevents deployment, and leaves the last deployed Pages site intact.
Failure diagnostics are uploaded from `.build/failed-build`. The retained site
will eventually show its stale forecast warning. Historical NPZ archives are
not copied into new artifacts; archive retention beyond the CSV summaries
would need a separate storage policy.

## Dependency updates

Edit the direct pins in `requirements.in`, then regenerate and test the complete
hash-locked dependency set using Python 3.11:

```sh
uv pip compile --python venv/bin/python --generate-hashes \
  --output-file requirements.txt requirements.in
python -m pip install --require-hashes -r requirements.txt
python -m unittest discover -s tests -v
python build_site.py --output .build/site
```

Regression tests cover deterministic field selection, projected/scanning grid
orientation, accumulation and cycle validation, millimeter-to-inch conversion,
full rendering, nested 1/2/3-day totals, subbasin chart traces, gage missing
values, HTTP range failures, and preservation of
the previous artifact after forecast failure. They require no live data.
