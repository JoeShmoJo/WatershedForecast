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

`--hours` accepts 1–36 (default 24); `--workers` controls download concurrency,
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

Only deterministic **one-hour surface APCP** messages are selected. Probability
fields and six-hour accumulations are excluded. The old six-hour boundary
subtraction is gone. Native Lambert conformal coordinates and alternating row
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

Each basin gets PNG/SVG maps, interactive HTML charts, and `hourly_precip.csv`
with UTC interval-ending timestamps. `status.json` records the actual model
initialization, forecast interval, source, and each gage's availability/latest
valid observation. The webpage marks forecasts older than 18 hours and
observations older than three hours as stale, including when updates stop.

## Gage availability

Observations are fetched independently before forecast retrieval. Missing-data
sentinels are removed. An unavailable gage is labeled explicitly and does not
block otherwise valid precipitation forecasts. Observations and their status
remain in failed-build diagnostics even when forecasting fails.

During live validation on November 11, 2026, Amboy, Hatchery, Underwood, and
EFL Heisson returned recent observations. Carson returned missing-data values,
Washougal returned no flow series, and both WA Ecology TXT downloads returned
HTTP 403 from their upstream application gateway. Those four sources are shown
as unavailable; this migration does not invent replacement gage IDs or
circumvent upstream access restrictions. Their availability may change.

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
full rendering, gage missing values, HTTP range failures, and preservation of
the previous artifact after forecast failure. They require no live data.
