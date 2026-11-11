"""Build a complete, validated Pages artifact without changing checked-in plots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

# Set writable caches before importing matplotlib through the plotting module.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely import make_valid

import WatershedForecast as plots
from gageDownload import download_gage
from nbm_grib import load_forecast
from write_index import build_index_html


def fetch_gage(gage):
    label = gage.get("name") or gage["site_id"]
    status = {"name": label, "service": gage["service"], "site_id": gage["site_id"]}
    try:
        series, meta = download_gage(gage, hours=24)
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if gage["type"].upper() == "FLOW":
            series = series[series >= 0]
        if series.empty:
            raise ValueError("No valid observations in the last 24 hours")
        latest = series.index.max()
        age = (pd.Timestamp.now(tz="UTC")-latest).total_seconds()/3600
        status.update(state="stale" if age > 3 else "ok", latest=latest.isoformat(), units=meta["units"])
        return (label, series, meta["units"]), status
    except Exception as e:
        logging.warning("Gage %s unavailable: %s", label, e)
        status.update(state="unavailable", error=str(e))
        return None, status


def load_basins(root, configs):
    geometries = {}
    for cfg in configs:
        path = root / cfg.basins_dir / cfg.name / "basin.geojson"
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Basin has no coordinate reference system: {path}")
        originals = frame.to_crs("EPSG:4326").geometry
        if not originals.is_valid.all():
            logging.warning("Repairing cached basin ring intersections in memory: %s", cfg.name)
        geometry = unary_union([make_valid(g) for g in originals])
        if geometry.geom_type == "GeometryCollection":
            geometry = unary_union([g for g in geometry.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid basin geometry: {path}")
        geometries[cfg.name] = geometry
    return geometries


def validate_site(site, configs, hours):
    for cfg in configs:
        folder = site / "assets" / cfg.name
        for name in ("area_precip_latest.png", "area_precip_latest.svg", "ts_precip_latest.html"):
            if not (folder/name).is_file() or (folder/name).stat().st_size < 100:
                raise RuntimeError(f"Missing generated output: {folder/name}")
        frame = pd.read_csv(folder/"hourly_precip.csv")
        if len(frame) != hours or not np.isfinite(frame["precipitation_inches"]).all() or (frame["precipitation_inches"] < 0).any():
            raise RuntimeError(f"Incomplete precipitation series: {cfg.name}")
    if not (site/"index.html").is_file():
        raise RuntimeError("Missing site index")


def build(root, output, cache, basin=None, hours=24, workers=4, upscale=2):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output in root.parents or output == root/"docs":
        raise ValueError("Use a separate output directory, not the checkout or tracked docs")
    if output.exists() and (not (output/"status.json").is_file() or output.is_symlink()):
        raise ValueError(f"Refusing to replace an unrecognized output directory: {output}")
    config = plots.load_config_file(root)
    if basin is not None and basin not in config:
        raise ValueError("--basin must name a configured parent basin")
    configs = plots.expand_run_configs(config, hours, upscale, True, root, Path("Basins"), only=basin)
    selected = {c.name.split('/')[0] for c in configs}
    parents = {name: info for name, info in config.items() if name in selected}
    geometries = load_basins(root, configs)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="watershed-", dir=output.parent))
    site = work/"docs"
    site.mkdir()
    status = {"generated_at": datetime.now(timezone.utc).isoformat(), "forecast": {"state": "pending"}, "gages": {}}
    try:
        # Observations are fetched and rendered before forecast retrieval; a
        # forecast failure cannot suppress observation checks or diagnostics.
        tasks = [(name, g) for name, info in parents.items() for g in info.get("gages", [])]
        with ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
            results = list(pool.map(lambda item: fetch_gage(item[1]), tasks))
        for name in parents:
            folder = site/"assets"/name
            folder.mkdir(parents=True, exist_ok=True)
            series, statuses = [], []
            for (parent, _), (data, item_status) in zip(tasks, results):
                if parent == name:
                    statuses.append(item_status)
                    if data is not None:
                        series.append(data)
            status["gages"][name] = statuses
            plots.plotly_ts_gages(series, f"{name} — observations (last 24h)", str(folder/"gage_latest.html"))
        bounds = unary_union(list(geometries.values())).bounds
        forecast = load_forecast(bounds, hours, cache, workers)
        status["forecast"] = {"state": "ok", **forecast.attrs,
                              "valid_start": str(forecast.time.values[0]), "valid_end": str(forecast.time.values[-1]), "hours": hours}
        # The GRIB adapter guarantees hourly mm; do NOT apply the retired
        # OPeNDAP six-hour boundary correction to these messages.
        arr = forecast.values/25.4
        times = pd.DatetimeIndex(forecast.time.values).tz_localize("UTC")
        series_by_basin = {}
        for cfg in configs:
            geometry = geometries[cfg.name]
            lat, lon, dy, dx = plots.build_fine_grid(forecast.lat.values, forecast.lon.values, geometry.bounds, upscale)
            mask, weights = plots.basin_mask_and_weights(geometry, lat, lon, True)
            if not mask.any() or weights.sum() == 0:
                raise ValueError(f"Basin contains no sampled grid points: {cfg.name}")
            hourly, cumulative = plots.interpolate_and_aggregate(arr, forecast.lat.values, forecast.lon.values, lat, lon, weights)
            hourly.index = times
            series_by_basin[cfg.name] = hourly
            folder = site/"assets"/cfg.name
            folder.mkdir(parents=True, exist_ok=True)
            outlines = {n.split('/', 1)[1]: g for n, g in geometries.items() if n.startswith(cfg.name+'/')}
            plots.plot_cumulative_map(geometry, lat, lon, dy, dx, mask, cumulative,
                                      f"{cfg.name}\nNBM {forecast.attrs['cycle']} — {hours}h precipitation",
                                      str(folder/"area_precip_latest"), outlines)
            pd.DataFrame({"valid_time_utc": times, "precipitation_inches": hourly.values}).to_csv(folder/"hourly_precip.csv", index=False)
            logging.info("Rendered %s: %.3f inches over %d hours", cfg.name, hourly.sum(), hours)
        for cfg in configs:
            subs = {n.split('/', 1)[1]: s for n, s in series_by_basin.items() if n.startswith(cfg.name+'/')}
            plots.plotly_ts_precip_multi(series_by_basin[cfg.name], subs,
                                         f"{cfg.name} — NBM {forecast.attrs['cycle']}",
                                         str(site/"assets"/cfg.name/"ts_precip_latest.html"))
        # The index writer expects a root containing Config.json and docs/.
        (work/"Config.json").write_text(json.dumps(parents))
        (site/"status.json").write_text(json.dumps(status, indent=2))
        (site/"index.html").write_text(build_index_html(work, "Watershed Forecast Viewer"))
        (site/".nojekyll").touch()
        validate_site(site, configs, hours)
        # Preserve the previous validated artifact until this build is complete.
        backup = work/"previous"
        if output.exists():
            output.rename(backup)
        try:
            site.rename(output)
        except Exception:
            if backup.exists():
                backup.rename(output)
            raise
        logging.info("Validated site written to %s", output)
        return status
    except Exception as e:
        status["forecast"] = {"state": "failed", "error": str(e)}
        (site/"status.json").write_text(json.dumps(status, indent=2))
        diagnostic = output.parent/"failed-build"
        if diagnostic.exists() and (not (diagnostic/"status.json").is_file() or diagnostic.is_symlink()):
            raise RuntimeError(f"Refusing to replace unrecognized diagnostics directory: {diagnostic}") from e
        if diagnostic.exists():
            shutil.rmtree(diagnostic)
        site.rename(diagnostic)
        logging.error("Build failed; previous site preserved. Diagnostics: %s", diagnostic)
        raise
    finally:
        shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(".build/site"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/nbm"))
    parser.add_argument("--basin", help="One parent basin (including its subbasins)")
    parser.add_argument("--all", action="store_true", help="Build all basins (the default)")
    parser.add_argument("--hours", type=int, default=24, choices=range(1, 37))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent network requests (maximum 4)")
    parser.add_argument("--upscale", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    try:
        build(args.root, args.output, args.cache, args.basin, args.hours, args.workers, args.upscale)
    except Exception:
        logging.exception("Site build failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
