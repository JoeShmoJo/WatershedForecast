#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WatershedForecast.py (single-axis version)

- Reads Config.json for named basins (state, lat, lon, optional USGS_id/location)
- Downloads NBM 1-hr precip via NOMADS OPeNDAP
- Boundary-hour correction for 6h accumulation edge hours
- Fine-grid bilinear upscaling + basin mask (center-in-polygon)
- Area-weighted basin average (cos(latitude))
- Single x-axis: majors at local midnights inside window (date labels),
  minors every 3 hours (HH:MM)
- Saves PNG+SVG under docs/assets/<BASIN_NAME>/ and updates *_latest.svg/png
- Optional USGS streamflow (00060) past 24h plot → usgs_flow_latest.[png|svg]
- CLI: --basin NAME | --all, plus --offline / timeouts / retries
"""

from __future__ import annotations

# --- non-interactive backend for CI/servers ---
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.dates import date2num, DateFormatter, HourLocator
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
from shapely.geometry import Point, Polygon, shape
from shapely.prepared import prep as prepare_polygon
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ----------------------------
# Configuration
# ----------------------------
@dataclass(frozen=True)
class RunConfig:
    name: str
    state: str
    lattitude: float
    longitude: float
    USGS_id: str = ""
    USGS_location: str = ""
    hours: int = 24
    upscale_factor: int = 4
    area_weighting: bool = True
    root: Path = Path(".")            # repo root (where Config.json lives)
    basins_dir: Path = Path("Basins") # cache dir for basin geojsons


def load_config_file(root: Path) -> Dict[str, Dict[str, float | str]]:
    cfg_path = root / "Config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# HTTP session with retries/backoff
# ----------------------------
def make_retrying_session(total=5, backoff=1.5, timeout=30):
    """
    Returns (session, timeout_seconds). Session retries on transient errors
    with exponential backoff.
    """
    retry = Retry(
        total=total, read=total, connect=total, status=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s, timeout


# ----------------------------
# Utilities: coordinates & grids
# ----------------------------
def coord_slice(coord_values: np.ndarray, vmin: float, vmax: float) -> slice:
    return slice(vmin, vmax) if coord_values[0] <= coord_values[-1] else slice(vmax, vmin)


def ensure_ascending(lat_vals, lon_vals, grid3d):
    lat = np.asarray(lat_vals)
    lon = np.asarray(lon_vals)
    arr = np.asarray(grid3d)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr = arr[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[:, :, ::-1]
    return lat, lon, arr


def build_index_weights_for_bilinear(lat_src, lon_src, lat_f, lon_f) -> Dict[str, np.ndarray]:
    lat0, lon0 = float(lat_src[0]), float(lon_src[0])
    dlat = float(np.median(np.diff(lat_src)))
    dlon = float(np.median(np.diff(lon_src)))
    jj = (lon_f - lon0) / dlon
    ii = (lat_f - lat0) / dlat
    j0 = np.floor(jj).astype(int)
    i0 = np.floor(ii).astype(int)
    j1 = np.clip(j0 + 1, 0, lon_src.size - 1)
    i1 = np.clip(i0 + 1, 0, lat_src.size - 1)
    j0 = np.clip(j0, 0, lon_src.size - 1)
    i0 = np.clip(i0, 0, lat_src.size - 1)
    fj = (jj - j0)
    fi = (ii - i0)
    return dict(i0=i0, i1=i1, j0=j0, j1=j1, fi=fi, fj=fj)


def bilinear_apply(grid2d: np.ndarray, idxw: Dict[str, np.ndarray]) -> np.ndarray:
    i0, i1, j0, j1, fi, fj = (idxw[k] for k in ("i0", "i1", "j0", "j1", "fi", "fj"))
    Q11 = grid2d[i0[:, None], j0[None, :]]
    Q21 = grid2d[i1[:, None], j0[None, :]]
    Q12 = grid2d[i0[:, None], j1[None, :]]
    Q22 = grid2d[i1[:, None], j1[None, :]]
    fi2 = fi.reshape(-1, 1)
    fj2 = fj.reshape(1, -1)
    return (1 - fi2) * (1 - fj2) * Q11 + fi2 * (1 - fj2) * Q21 + (1 - fi2) * fj2 * Q12 + fi2 * fj2 * Q22


# ----------------------------
# Data access
# ----------------------------
def find_latest_nbm_opendap_url() -> str:
    base = "https://nomads.ncep.noaa.gov/dods/blend"
    cycles = ["18z", "12z", "06z", "00z"]
    for day_offset in range(0, 3):
        date_str = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y%m%d")
        dds = None
        for cyc in cycles:
            dds_url = f"{base}/blend{date_str}/blend_1hr_{cyc}.dds"
            try:
                r = requests.get(dds_url, timeout=6)
                if r.status_code == 200 and "Dataset" in r.text[:200]:
                    dds = f"{base}/blend{date_str}/blend_1hr_{cyc}"
                    break
            except requests.RequestException:
                continue
        if dds:
            return dds
    raise RuntimeError("No recent NBM 1-hr OPeNDAP dataset found on NOMADS (last 3 days).")


def open_nbm_subset(opendap_url: str, basin: Polygon, hours: int) -> Tuple[xr.DataArray, Tuple[float, float, float, float]]:
    ds = xr.open_dataset(opendap_url, engine="netcdf4", cache=False)
    minx, miny, maxx, maxy = basin.bounds
    pad = 0.05
    lat_slice = coord_slice(ds["lat"].values, miny - pad, maxy + pad)
    lon_slice = coord_slice(ds["lon"].values, minx - pad, maxx + pad)
    qpf = ds["apcpsfc"].sel(lat=lat_slice, lon=lon_slice).isel(time=slice(0, hours)).load()
    ds.close()
    return qpf, (minx, miny, maxx, maxy)


# ---- USGS NWIS (Instantaneous Values) for discharge 00060 (past 24h) ----
def get_usgs_flow(usgs_id: str, end_time: datetime) -> pd.Series:
    """
    Fetch past 24h discharge (00060) for a site.
    Returns a Series indexed in America/Los_Angeles (tz-aware).
    """
    # Ensure end_time is UTC-aware
    # make end_time current time 
    end_time = datetime.now(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = end_time.astimezone(timezone.utc)

    start_time = end_time - timedelta(hours=24)

    # Use Z (UTC) to avoid ambiguity
    url = (
        "https://waterservices.usgs.gov/nwis/iv/?format=json"
        f"&sites={usgs_id}"
        "&parameterCd=00060"
        f"&startDT={start_time.strftime('%Y-%m-%dT%H:%MZ')}"
        f"&endDT={end_time.strftime('%Y-%m-%dT%H:%MZ')}"
        "&siteStatus=all"
    )

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"USGS request failed for site {usgs_id}: {e}") from e

    data = r.json()
    ts = data.get("value", {}).get("timeSeries", [])
    if not ts:
        raise ValueError(f"USGS site {usgs_id}: no timeSeries in response.")

    vals = ts[0].get("values", [])
    if not vals or not vals[0].get("value"):
        raise ValueError(f"USGS site {usgs_id}: empty values array.")

    recs = vals[0]["value"]
    times = [pd.to_datetime(rec["dateTime"], utc=True) for rec in recs]  # tz-aware UTC
    flows = [float(rec["value"]) for rec in recs]

    ser = pd.Series(flows, index=pd.DatetimeIndex(times)).sort_index()
    ser = ser.tz_convert("America/Los_Angeles")

    logging.info("USGS %s: fetched %d points from %s to %s (local).",
                 usgs_id, len(ser),
                 ser.index.min().strftime('%Y-%m-%d %H:%M %Z') if len(ser) else "n/a",
                 ser.index.max().strftime('%Y-%m-%d %H:%M %Z') if len(ser) else "n/a")
    return ser


# ----------------------------
# Basin geometry (cache-first, legacy fallback, optional offline)
# ----------------------------
def get_watershed_basin(cfg: RunConfig, offline: bool = False, streamstats_timeout: int = 30, streamstats_retries: int = 5) -> Polygon:
    """
    Cache at Basins/<name>/basin.geojson
    Legacy fallback: Basins/basin.geojson if per-basin file not present.
    If offline=True, will NOT call StreamStats; requires a cached geojson.
    """
    basin_dir = cfg.root / cfg.basins_dir / cfg.name
    basin_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = basin_dir / "basin.geojson"

    # Preferred cache
    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
        return gdf.geometry.iloc[0]

    # Legacy fallback
    legacy_geo = cfg.root / cfg.basins_dir / "basin.geojson"
    if legacy_geo.exists():
        logging.warning("Using legacy Basins/basin.geojson for %s (migrating to per-basin cache).", cfg.name)
        gdf = gpd.read_file(legacy_geo)
        geom = gdf.geometry.iloc[0]
        gpd.GeoDataFrame(index=[0], geometry=[geom], crs="EPSG:4326").to_file(geojson_path, driver="GeoJSON")
        return geom

    if offline:
        raise RuntimeError(
            f"No cached basin found for {cfg.name} and offline mode is enabled.\n"
            f"Expected at: {geojson_path}"
        )

    # Fetch from StreamStats with retries/backoff
    url = (
        "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson?"
        f"rcode={cfg.state}&xlocation={cfg.longitude}&ylocation={cfg.lattitude}&crs=4326"
        "&includeparameters=false&includeflowtypes=false&includefeatures=true&simplify=true"
    )
    session, timeout = make_retrying_session(total=streamstats_retries, backoff=1.5, timeout=streamstats_timeout)
    logging.info("Requesting StreamStats basin for %s…", cfg.name)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch basin from StreamStats after retries (timeout={timeout}s, retries={streamstats_retries}). "
            f"Consider running offline with a cached geojson. Original error: {e}"
        ) from e

    data = r.json()
    geom = None
    for fc in data.get("featurecollection", []):
        if fc.get("name") == "globalwatershed":
            geom = shape(fc["feature"]["features"][0]["geometry"])
            break
    if geom is None:
        raise ValueError("Watershed polygon not found from StreamStats response.")
    gpd.GeoDataFrame(index=[0], geometry=[geom], crs="EPSG:4326").to_file(geojson_path, driver="GeoJSON")
    return geom


# ----------------------------
# Transformations & aggregation
# ----------------------------
def boundary_correct_3d(arr_in_inch: np.ndarray, time_vals) -> np.ndarray:
    arr = np.asarray(arr_in_inch, dtype=float).copy()
    nt = arr.shape[0]
    for t in range(nt):
        hr = int(pd.to_datetime(time_vals[t]).hour)
        if hr in (0, 6, 12, 18):
            t0 = max(0, t - 5)
            arr[t, :, :] = np.maximum(0.0, arr[t, :, :] - np.nansum(arr[t0:t, :, :], axis=0))
    return arr


def build_fine_grid(lat_src: np.ndarray, lon_src: np.ndarray, bbox: Tuple[float, float, float, float], upscale: int):
    minx, miny, maxx, maxy = bbox
    dlat_src = float(np.median(np.diff(lat_src)))
    dlon_src = float(np.median(np.diff(lon_src)))
    dlat_f = dlat_src / upscale
    dlon_f = dlon_src / upscale
    lon_f = np.arange(minx, maxx + 1e-12, dlon_f)
    lat_f = np.arange(miny, maxy + 1e-12, dlat_f)
    lon_f = lon_f[(lon_f >= lon_src.min()) & (lon_f <= lon_src.max())]
    lat_f = lat_f[(lat_f >= lat_src.min()) & (lat_f <= lat_src.max())]
    return lat_f, lon_f, dlat_f, dlon_f


def basin_mask_and_weights(basin: Polygon, lat_f: np.ndarray, lon_f: np.ndarray, area_weighting: bool) -> Tuple[np.ndarray, np.ndarray]:
    YY, XX = np.meshgrid(lat_f, lon_f, indexing="ij")
    centers = np.column_stack([XX.ravel(), YY.ravel()])
    P = prepare_polygon(basin)
    mask_flat = np.fromiter((P.contains(Point(x, y)) for x, y in centers), dtype=bool, count=centers.shape[0])
    mask = mask_flat.reshape(YY.shape)
    if area_weighting:
        row_w = np.cos(np.deg2rad(lat_f))
        W = np.repeat(row_w[:, None], lon_f.size, axis=1) * mask
    else:
        W = mask.astype(float)
    return mask, W


def interpolate_and_aggregate(arr3d: np.ndarray,
                              lat_src: np.ndarray,
                              lon_src: np.ndarray,
                              lat_f: np.ndarray,
                              lon_f: np.ndarray,
                              weights: np.ndarray) -> Tuple[pd.Series, np.ndarray]:
    idxw = build_index_weights_for_bilinear(lat_src, lon_src, lat_f, lon_f)
    ny, nx = len(lat_f), len(lon_f)
    fine_cum = np.zeros((ny, nx), dtype=float)
    hourly_vals = []
    for t in range(arr3d.shape[0]):
        fine_t = bilinear_apply(arr3d[t, :, :], idxw)
        fine_cum += np.where(np.isfinite(fine_t), fine_t, 0.0)
        num = np.nansum(fine_t * weights)
        den = np.nansum(weights)
        hourly_vals.append(0.0 if den == 0 else num / den)
    return pd.Series(hourly_vals), fine_cum


# ----------------------------
# Plotting
# ----------------------------
def wpc_qpf_colormap_extended():
    levels = [0.00, 0.01, 0.10, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    colors = [
        "#ffffff", "#c8facc", "#7be37c", "#c7f266", "#ffe66b", "#ffae42",
        "#ff6d3a", "#e53228", "#b3175b", "#6a1b9a", "#4a148c", "#311b92", "#1a237e"
    ]
    cmap = ListedColormap(colors, name="wpc_qpf_ext")
    norm = BoundaryNorm(levels, cmap.N, extend="max")
    return cmap, norm, levels


def style_inrange_day_majors_3h_minors(ax, tindex, hour_step=3):
    """
    Single-axis styling:
      - Major ticks: local midnights strictly inside the data window (date labels)
      - Minor ticks: every `hour_step` hours (HH:MM labels)
    """
    # Guard: ensure tz-aware
    if getattr(tindex, "tz", None) is None:
        tindex = tindex.tz_localize("UTC")

    tz = tindex.tz
    tmin, tmax = tindex.min(), tindex.max()

    # majors at midnights strictly inside window
    dn_start, dn_end = tmin.normalize(), tmax.normalize()
    all_midnights = pd.date_range(dn_start, dn_end + pd.Timedelta(days=1), freq="D", tz=tz)
    mids_in = all_midnights[(all_midnights > tmin) & (all_midnights < tmax)]

    if len(mids_in):
        ax.xaxis.set_major_locator(FixedLocator(date2num([m.to_pydatetime() for m in mids_in])))
        ax.xaxis.set_major_formatter(DateFormatter('%a %b %d', tz=tz))
    else:
        ax.xaxis.set_major_locator(NullLocator())
        ax.xaxis.set_major_formatter(NullFormatter())

    # minors every N hours
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(0, 24, hour_step), tz=tz))
    ax.xaxis.set_minor_formatter(DateFormatter('%H:%M', tz=tz))

    ax.tick_params(axis='x', which='major', length=8, width=1.2, pad=6)
    ax.tick_params(axis='x', which='minor', length=4, width=0.8, pad=2)

    for lbl in ax.get_xticklabels(which='major'):
        lbl.set_rotation(0); lbl.set_fontsize(10)
    for lbl in ax.get_xticklabels(which='minor'):
        lbl.set_rotation(0); lbl.set_fontsize(9)

    ax.grid(True, which='major', axis='x', alpha=0.15, linewidth=0.8)
    ax.grid(True, which='minor', axis='x', alpha=0.08, linewidth=0.6)


def plot_cumulative_map(basin: Polygon,
                        lat_f: np.ndarray,
                        lon_f: np.ndarray,
                        dlat_f: float,
                        dlon_f: float,
                        mask: np.ndarray,
                        fine_cum: np.ndarray,
                        title: str,
                        save_basepath: str | None = None):
    cmap, norm, levels = wpc_qpf_colormap_extended()
    minx, miny, maxx, maxy = basin.bounds
    fig, ax = plt.subplots(figsize=(10, 6))
    gpd.GeoSeries([basin], crs="EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)
    for iy, cy in enumerate(lat_f):
        for ix, cx in enumerate(lon_f):
            if not mask[iy, ix]:
                continue
            val = fine_cum[iy, ix]
            rect = Rectangle(
                (float(cx) - 0.5 * dlon_f, float(cy) - 0.5 * dlat_f),
                dlon_f, dlat_f,
                facecolor=cmap(norm(val)),
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )
            ax.add_patch(rect)
    gpd.GeoSeries([basin], crs="EPSG:4326").plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0, zorder=4)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Cumulative precipitation (in)")
    cb.set_ticks(levels)
    cb.ax.set_yticklabels([f"{lv:g}" for lv in levels])
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    if save_basepath:
        fig.savefig(f"{save_basepath}.png", dpi=160, bbox_inches="tight")
        fig.savefig(f"{save_basepath}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_hourly_and_cumulative_local(hourly_in_utc: pd.Series, title: str, save_basepath: str | None = None):
    hourly_local = hourly_in_utc.copy()
    hourly_local.index = (
        pd.DatetimeIndex(hourly_local.index)
        .tz_localize("UTC")
        .tz_convert("America/Los_Angeles")
    )
    cumulative_local = hourly_local.cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(hourly_local.index, hourly_local.values, width=pd.Timedelta("45min"),
            label="Hourly Precip (in)", color="tab:blue")
    ax1.plot(cumulative_local.index, cumulative_local.values, marker="o",
             label="Cumulative Precip (in)", color="tab:orange")

    style_inrange_day_majors_3h_minors(ax1, hourly_local.index, hour_step=3)

    ax1.set_xlim(hourly_local.index.min(), hourly_local.index.max())
    ax1.margins(x=0.01)

    ax1.set_ylabel("Precipitation (in)")
    ax1.set_xlabel("Local time")
    ax1.set_title(title)
    ax1.legend()

    if save_basepath:
        fig.savefig(f"{save_basepath}.png", dpi=160, bbox_inches="tight")
        fig.savefig(f"{save_basepath}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_usgs_flow(flow_series: pd.Series | None,
                   title: str,
                   save_basepath: str | None = None,
                   hour_step: int = 3):
    """Plot USGS flow if available; otherwise save a blank/placeholder canvas."""
    # Create figure up front
    fig, ax = plt.subplots(figsize=(10, 5))

    # Nothing to plot?
    empty = (
        flow_series is None
        or len(flow_series) == 0
        or len(flow_series.dropna()) == 0
    )

    if empty:
        # Clean blank canvas with a subtle message
        ax.set_axis_off()
        fig.suptitle(title, y=0.98, fontsize=14)
        fig.text(0.5, 0.50, "No recent USGS data available",
                 ha="center", va="center", fontsize=12, alpha=0.6)
    else:
        # Ensure tz-aware + local for tick styling
        idx = flow_series.index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
            flow_series.index = idx
        flow_series = flow_series.tz_convert("America/Los_Angeles")

        ax.plot(flow_series.index, flow_series.values,
                marker="o", color="tab:green", label="Discharge (cfs)")

        style_inrange_day_majors_3h_minors(ax, flow_series.index, hour_step=hour_step)
        ax.set_xlim(flow_series.index.min(), flow_series.index.max())
        ax.margins(x=0.01)

        ax.set_ylabel("Discharge (cfs)")
        ax.set_xlabel("Local time")
        ax.set_title(title)
        ax.legend()

    # Save no matter what so index.html can always reference the files
    if save_basepath:
        fig.savefig(f"{save_basepath}.png", dpi=160, bbox_inches="tight")
        fig.savefig(f"{save_basepath}.svg", bbox_inches="tight")
    plt.close(fig)



# ----------------------------
# Orchestration
# ----------------------------
def run_one(cfg: RunConfig, offline: bool = False, streamstats_timeout: int = 30, streamstats_retries: int = 5):
    # 1) Basin
    basin = get_watershed_basin(cfg, offline=offline, streamstats_timeout=streamstats_timeout, streamstats_retries=streamstats_retries)
    logging.info("Basin ready: %s", cfg.name)

    # 2) Forecast
    url = find_latest_nbm_opendap_url()
    logging.info("Using forecast: %s", url)
    qpf, bbox = open_nbm_subset(url, basin, cfg.hours)

    # 3) mm -> in, boundary fix
    times_utc = pd.to_datetime(qpf.time.values)
    coarse_in = (qpf.values / 25.4).astype("float64")
    coarse_in_corr = boundary_correct_3d(coarse_in, qpf.time.values)

    # 4) Ascending coords
    lat_src, lon_src, arr3d = ensure_ascending(qpf.lat.values, qpf.lon.values, coarse_in_corr)

    # 5) Fine grid + weights
    lat_f, lon_f, dlat_f, dlon_f = build_fine_grid(lat_src, lon_src, bbox, cfg.upscale_factor)
    mask, weights = basin_mask_and_weights(basin, lat_f, lon_f, cfg.area_weighting)

    # 6) Interpolate + aggregate
    hourly_series_in, fine_cum = interpolate_and_aggregate(arr3d, lat_src, lon_src, lat_f, lon_f, weights)
    hourly_series_in.index = times_utc  # UTC

    # 7) Output paths for GitHub Pages
    out_dir = cfg.root / "docs" / "assets" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y%m%d_%H%M")
    map_base = str(out_dir / f"cumulative_map_{stamp}")
    ts_base  = str(out_dir / f"hourly_series_{stamp}")

    # 8) Save plots (precip)
    plot_cumulative_map(
        basin, lat_f, lon_f, dlat_f, dlon_f, mask, fine_cum,
        title=f"{cfg.name} — Basin cumulative precipitation (fine grid x{cfg.upscale_factor})",
        save_basepath=map_base,
    )
    plot_hourly_and_cumulative_local(
        hourly_series_in,
        title=f"{cfg.name} Precipitation (Basin Avg; fine grid x{cfg.upscale_factor}) — local time",
        save_basepath=ts_base,
    )

    # 8.5) USGS streamflow (optional per Config.json)
    if cfg.USGS_id:
        try:
            end_time = hourly_series_in.index.max()
            if isinstance(end_time, np.datetime64):
                end_time = pd.to_datetime(end_time)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            flow_series = get_usgs_flow(cfg.USGS_id, end_time.to_pydatetime())
            flow_base = str(out_dir / f"usgs_flow_{stamp}")
            plot_usgs_flow(
                flow_series,
                title=f"{cfg.name} USGS Streamflow at {cfg.USGS_location} (Site {cfg.USGS_id}) — local time",
                save_basepath=flow_base,
            )
            # publish "latest"
            shutil.copyfile(f"{flow_base}.svg", out_dir / "usgs_flow_latest.svg")
            shutil.copyfile(f"{flow_base}.png", out_dir / "usgs_flow_latest.png")
            logging.info("Updated latest USGS flow symlikes in %s", out_dir)
        except Exception as e:
            logging.error("USGS flow step failed for %s (%s): %s", cfg.name, cfg.USGS_id, e)
    else:
        # create dummy plot that is a blank rectangle with "No Associated Gage or Gage Data"
        flow_base = str(out_dir / f"usgs_flow_{stamp}")
        plot_usgs_flow(
            None,
            title=f"{cfg.name} USGS Streamflow — No Associated Gage or Gage Data",
            save_basepath=flow_base,
        )

    # 9) Update stable "latest" copies (good for docs/index.html)
    shutil.copyfile(f"{map_base}.svg", out_dir / "cumulative_map_latest.svg")
    shutil.copyfile(f"{map_base}.png", out_dir / "cumulative_map_latest.png")
    shutil.copyfile(f"{ts_base}.svg",  out_dir / "hourly_series_latest.svg")
    shutil.copyfile(f"{ts_base}.png",  out_dir / "hourly_series_latest.png")

    logging.info("Saved outputs to %s", out_dir)


def run_all(root: Path, hours: int = 24, upscale: int = 4, area_weighting: bool = True,
            basins_dir: Path = Path("Basins"),
            offline: bool = False, streamstats_timeout: int = 30, streamstats_retries: int = 5):
    cfg_json = load_config_file(root)
    for name, info in cfg_json.items():
        cfg = RunConfig(
            name=name,
            state=str(info["state"]),
            lattitude=float(info["latitude"]),
            longitude=float(info["longitude"]),
            USGS_id=str(info.get("USGS_id", "")),
            USGS_location=str(info.get("USGS_location", "")),
            hours=hours,
            upscale_factor=upscale,
            area_weighting=area_weighting,
            root=root,
            basins_dir=basins_dir,
        )
        try:
            run_one(cfg, offline=offline, streamstats_timeout=streamstats_timeout, streamstats_retries=streamstats_retries)
        except Exception as e:
            logging.exception("Failed for %s: %s", name, e)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NBM basin precipitation forecast plots → docs/assets for GitHub Pages.")
    parser.add_argument("--root", type=str, default=".", help="Repo root (where Config.json lives). Default: current dir")
    parser.add_argument("--basin", type=str, default=None, help="Run a single basin by name (must match a Config.json key)")
    parser.add_argument("--all", action="store_true", help="Run all basins in Config.json")
    parser.add_argument("--hours", type=int, default=24, help="Forecast hours to use (from t=0). Default: 24")
    parser.add_argument("--upscale", type=int, default=4, help="Fine-grid upscale factor (1–4). Default: 4")
    parser.add_argument("--no-area-weighting", action="store_true", help="Disable cos(lat) area weighting")
    parser.add_argument("--basins-dir", type=str, default="Basins", help="Folder where basin caches live. Default: Basins")
    parser.add_argument("--offline", action="store_true", help="Do not call StreamStats; require cached Basins/<name>/basin.geojson.")
    parser.add_argument("--streamstats-timeout", type=int, default=30, help="Per-request timeout for StreamStats (seconds).")
    parser.add_argument("--streamstats-retries", type=int, default=5, help="Number of retries for StreamStats.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    basins_dir = Path(args.basins_dir)

    if args.all:
        run_all(root=root,
                hours=args.hours,
                upscale=args.upscale,
                area_weighting=(not args.no_area_weighting),
                basins_dir=basins_dir,
                offline=args.offline,
                streamstats_timeout=args.streamstats_timeout,
                streamstats_retries=args.streamstats_retries)
        return

    if not args.basin:
        parser.error("Specify --basin <NAME> or use --all")

    cfg_json = load_config_file(root)
    if args.basin not in cfg_json:
        raise SystemExit(f"Basin {args.basin!r} not found in Config.json keys: {list(cfg_json.keys())}")

    info = cfg_json[args.basin]
    cfg = RunConfig(
        name=args.basin,
        state=str(info["state"]),
        lattitude=float(info["latitude"]),
        longitude=float(info["longitude"]),
        USGS_id=str(info.get("USGS_id", "")),
        USGS_location=str(info.get("USGS_location", "")),
        hours=args.hours,
        upscale_factor=args.upscale,
        area_weighting=(not args.no_area_weighting),
        root=root,
        basins_dir=basins_dir,
    )
    run_one(cfg, offline=args.offline, streamstats_timeout=args.streamstats_timeout, streamstats_retries=args.streamstats_retries)


if __name__ == "__main__":
    main()
