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
from nbm_grib import load_forecast, forecast_steps
from write_index import build_index_html


def fetch_gage(gage):
    label = (gage.get("name") or gage["site_id"]).replace("_", " ")
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


def validate_site(site, configs, hours, subdir=""):
    expected = forecast_steps(hours)
    for cfg in configs:
        folder = site / "assets" / cfg.name / subdir
        for name in ("area_precip_latest.png", "area_precip_latest.svg", "ts_precip_latest.html"):
            if not (folder/name).is_file() or (folder/name).stat().st_size < 100:
                raise RuntimeError(f"Missing generated output: {folder/name}")
        frame = pd.read_csv(folder/"precipitation.csv")
        ends = np.cumsum(frame["interval_hours"].to_numpy())
        if (len(frame) != len(expected) or list(zip(ends, frame["interval_hours"])) != expected
                or not np.isfinite(frame["precipitation_inches"]).all()
                or (frame["precipitation_inches"] < 0).any()):
            raise RuntimeError(f"Incomplete precipitation intervals: {cfg.name} ({hours}h)")
    if not (site/"index.html").is_file():
        raise RuntimeError("Missing site index")


def build(root, output, cache, basin=None, hours=24, workers=4, upscale=2, days=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output in root.parents or output == root/"docs":
        raise ValueError("Use a separate output directory, not the checkout or tracked docs")
    if output.exists() and (not (output/"status.json").is_file() or output.is_symlink()):
        raise ValueError(f"Refusing to replace an unrecognized output directory: {output}")
    horizons = sorted({day*24 for day in days}) if days else [hours]
    hours = max(horizons)
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
            plots.plotly_ts_gages(series, f"{name.replace('_', ' ')} — observations (last 24h)", str(folder/"gage_latest.html"))
        bounds = unary_union(list(geometries.values())).bounds
        forecast = load_forecast(bounds, hours, cache, workers)
        interval_hours = (forecast["interval_hours"].values if "interval_hours" in forecast.coords
                          else np.ones(forecast.sizes["time"], dtype=int))
        times = pd.DatetimeIndex(forecast.time.values).tz_localize("UTC")
        elapsed = np.cumsum(interval_hours)
        status["forecast"] = {"state": "ok", **forecast.attrs, "hours": horizons[0],
                              "available_hours": horizons, "periods": {}}
        for horizon in horizons:
            selected_intervals = elapsed <= horizon
            durations = interval_hours[selected_intervals]
            period_times = times[selected_intervals]
            if durations.sum() != horizon:
                raise ValueError(f"Forecast does not cover {horizon} hours exactly")
            subdir = "" if horizon == horizons[0] else f"{horizon}h"
            period = {"hours": horizon, "valid_start": (period_times[0]-pd.Timedelta(hours=int(durations[0]))).isoformat(),
                      "valid_end": period_times[-1].isoformat(), "assets": {},
                      "interval_note": ("Bars show hourly precipitation totals." if durations.max() == 1 else
                                        "Bars show hourly totals for day 1, then six-hour totals. Lines show cumulative precipitation.")}
            status["forecast"]["periods"][str(horizon)] = period
            # Each message is already an explicit, nonoverlapping accumulation.
            # Never apply the retired OPeNDAP boundary correction.
            arr = forecast.values[selected_intervals]/25.4
            series_by_basin = {}
            for cfg in configs:
                geometry = geometries[cfg.name]
                lat, lon, dy, dx = plots.build_fine_grid(forecast.lat.values, forecast.lon.values, geometry.bounds, upscale)
                mask, weights = plots.basin_mask_and_weights(geometry, lat, lon, True)
                if not mask.any() or weights.sum() == 0:
                    raise ValueError(f"Basin contains no sampled grid points: {cfg.name}")
                precip, cumulative = plots.interpolate_and_aggregate(arr, forecast.lat.values, forecast.lon.values, lat, lon, weights)
                precip.index = period_times
                series_by_basin[cfg.name] = precip
                folder = site/"assets"/cfg.name/subdir
                folder.mkdir(parents=True, exist_ok=True)
                outlines = {n.split('/', 1)[1]: g for n, g in geometries.items() if n.startswith(cfg.name+'/')}
                plots.plot_cumulative_map(geometry, lat, lon, dy, dx, mask, cumulative,
                                          f"{cfg.name.replace('_', ' ')} · {horizon}h precipitation\n"
                                          f"{(period_times[0]-pd.Timedelta(hours=int(durations[0]))).tz_convert('America/Los_Angeles'):%b %d, %I%p}"
                                          f" – {period_times[-1].tz_convert('America/Los_Angeles'):%b %d, %I%p %Z}",
                                          str(folder/"area_precip_latest"), outlines)
                frame = pd.DataFrame({"valid_time_utc": period_times, "interval_hours": durations,
                                      "precipitation_inches": precip.values})
                frame.to_csv(folder/"precipitation.csv", index=False)
                if durations.max() == 1:
                    frame.to_csv(folder/"hourly_precip.csv", index=False)
                period["assets"][cfg.name] = {
                    "map": (folder/"area_precip_latest.png").relative_to(site).as_posix(),
                    "series": (folder/"ts_precip_latest.html").relative_to(site).as_posix(),
                    "csv": (folder/"precipitation.csv").relative_to(site).as_posix(),
                    "total_inches": float(precip.sum())}
                logging.info("Rendered %s (%dh): %.3f inches", cfg.name, horizon, precip.sum())
            for cfg in configs:
                subs = {n.split('/', 1)[1]: values for n, values in series_by_basin.items() if n.startswith(cfg.name+'/')}
                plots.plotly_ts_precip_multi(series_by_basin[cfg.name], subs,
                                             f"{cfg.name.replace('_', ' ')} — {horizon}h NBM forecast",
                                             str(site/"assets"/cfg.name/subdir/"ts_precip_latest.html"),
                                             base_label="Basin average",
                                             interval_hours=durations)
        default_period = status["forecast"]["periods"][str(horizons[0])]
        status["forecast"].update({key: default_period[key] for key in ("valid_start", "valid_end")})
        # The index writer expects a root containing Config.json and docs/.
        (work/"Config.json").write_text(json.dumps(parents))
        (site/"status.json").write_text(json.dumps(status, indent=2))
        (site/"index.html").write_text(build_index_html(work, "Watershed Forecast Viewer"))
        (site/".nojekyll").touch()
        for horizon in horizons:
            validate_site(site, configs, horizon, "" if horizon == horizons[0] else f"{horizon}h")
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
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--days", type=int, nargs="+", choices=[1, 2, 3], help="Available day controls (default: 1 2 3)")
    duration.add_argument("--hours", type=int, choices=list(range(1, 37))+[48, 72], help="Build one custom duration instead")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent network requests (maximum 4)")
    parser.add_argument("--upscale", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    try:
        days = args.days or ([1, 2, 3] if args.hours is None else None)
        build(args.root, args.output, args.cache, args.basin, args.hours or 24, args.workers, args.upscale, days=days)
    except Exception:
        logging.exception("Site build failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
