#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WatershedForecast.py — parent & subbasin plotting

What it does (per run item):
- Writes stamped outputs into that item's own folder:
    Parent:   docs/assets/<Parent>/
    Subbasin: docs/assets/<Parent>/<Sub>/
- Updates LATEST pointers only within that same folder, except:
    • gage_latest.html is parent-only (in docs/assets/<Parent>/).

Parent outputs (in docs/assets/<Parent>/):
  - area_precip_<stamp>.png/.svg (+ area_precip_latest.*)
  - ts_precip_<stamp>.html       (+ ts_precip_latest.html)  [combined parent + subbasins]
  - gage_<stamp>.html            (+ gage_latest.html)
  - archive/<Parent>_<stamp>.npz (parent only)

Subbasin outputs (in docs/assets/<Parent>/<Sub>/):
  - area_precip_<stamp>.png/.svg (+ area_precip_latest.*)
  - ts_precip_<stamp>.html       (+ ts_precip_latest.html)  [that subbasin only]

Behavior:
- Parent ts_precip: grouped hourly BARS + cumulative LINES for parent + each subbasin.
- Subbasin ts_precip: grouped hourly BARS + cumulative LINE for that single subbasin.
- Bars & lines share colors per series.
- Parent area map draws subbasin outlines; subbasin maps do not.
- Pruning: keep current + previous stamp in each folder (does not touch *latest).

Requires:
- Config.json (parents, optional subbasins, optional gages under parents)
- gageDownload.py (for gage plots)
"""

from __future__ import annotations

# --- non-interactive backend for CI/servers (for Matplotlib maps) ---
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
from typing import Dict, Tuple, List

import warnings
from xarray.coding.times import SerializationWarning

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon, shape
from shapely.prepared import prep as prepare_polygon
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Plotly (for timeseries)
try:
    import plotly.graph_objs as go
    from plotly.offline import plot as plotly_plot
    _PLOTLY_OK = True
except Exception as _e:
    logging.warning("Plotly import failed; timeseries HTML will be skipped: %s", _e)
    _PLOTLY_OK = False

# gage helpers (your separate file in repo root)
try:
    from gageDownload import download_gage
    _GAGE_OK = True
except Exception as _e:
    logging.warning("gageDownload import failed; gage plots disabled: %s", _e)
    _GAGE_OK = False


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ----------------------------
# Index writing
# ----------------------------


def _write_index_after_run(root: Path, title: str = "Watershed Forecast Viewer") -> None:
    try:
        # Prefer importing the function so we don't spawn another process
        import write_index
        html = write_index.build_index_html(root, title)
        out = root / "docs" / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logging.info("Wrote index: %s", out)
    except Exception as e:
        logging.warning("Could not write docs/index.html via write_index.py: %s", e)

# ----------------------------
# Configuration
# ----------------------------
@dataclass(frozen=True)
class RunConfig:
    name: str
    state: str
    lattitude: float
    longitude: float
    USGS_id: str = ""          # kept for legacy; gages come from Config['...']['gages']
    USGS_location: str = ""
    hours: int = 24
    upscale_factor: int = 4
    area_weighting: bool = True
    root: Path = Path(".")
    basins_dir: Path = Path("Basins")


def load_config_file(root: Path) -> Dict[str, dict]:
    cfg_path = root / "Config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_run_configs(cfg_json: Dict[str, dict],
                       hours: int,
                       upscale: int,
                       area_weighting: bool,
                       root: Path,
                       basins_dir: Path,
                       only: str | None = None,
                       include_subbasins: bool = True) -> list[RunConfig]:
    def mk_cfg(name: str, info: dict) -> RunConfig:
        return RunConfig(
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

    cfgs: list[RunConfig] = []

    if only and "/" in only:
        parent, sub = only.split("/", 1)
        pinfo = cfg_json.get(parent)
        if not pinfo:
            raise SystemExit(f"Parent basin {parent!r} not found in Config.json.")
        sb = pinfo.get("subbasins", {}).get(sub)
        if not sb:
            raise SystemExit(f"Subbasin {sub!r} not found under {parent!r}.")
        sb_info = {"state": sb["state"], "latitude": sb["latitude"], "longitude": sb["longitude"]}
        cfgs.append(mk_cfg(f"{parent}/{sub}", sb_info))
        return cfgs

    if only and only in cfg_json:
        pinfo = cfg_json[only]
        cfgs.append(mk_cfg(only, pinfo))
        if include_subbasins and "subbasins" in pinfo and isinstance(pinfo["subbasins"], dict):
            for sub_name, sb in pinfo["subbasins"].items():
                sb_info = {"state": sb["state"], "latitude": sb["latitude"], "longitude": sb["longitude"]}
                cfgs.append(mk_cfg(f"{only}/{sub_name}", sb_info))
        return cfgs

    if only and only not in cfg_json:
        raise SystemExit(f"Basin {only!r} not found in Config.json keys: {list(cfg_json.keys())}")

    # all
    for parent, pinfo in cfg_json.items():
        cfgs.append(mk_cfg(parent, pinfo))
        if "subbasins" in pinfo and isinstance(pinfo["subbasins"], dict):
            for sub_name, sb in pinfo["subbasins"].items():
                sb_info = {"state": sb["state"], "latitude": sb["latitude"], "longitude": sb["longitude"]}
                cfgs.append(mk_cfg(f"{parent}/{sub_name}", sb_info))

    return cfgs


# ----------------------------
# HTTP session with retries
# ----------------------------
def make_retrying_session(total=5, backoff=1.5, timeout=30):
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
# Data access (NBM)
# ----------------------------
def find_latest_nbm_opendap_url() -> str:
    base = "https://nomads.ncep.noaa.gov/dods/blend"
    cycles = ["18z", "12z", "06z", "00z"]
    for day_offset in range(0, 3):
        date_str = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y%m%d")
        for cyc in cycles:
            dds_url = f"{base}/blend{date_str}/blend_1hr_{cyc}.dds"
            try:
                r = requests.get(dds_url, timeout=6)
                if r.status_code == 200 and "Dataset" in r.text[:200]:
                    return f"{base}/blend{date_str}/blend_1hr_{cyc}"
            except requests.RequestException:
                continue
    raise RuntimeError("No recent NBM 1-hr OPeNDAP dataset found on NOMADS (last 3 days).")


def get_locked_nbm_url() -> str:
    forced = os.environ.get("NBM_URL")
    if forced:
        logging.info("NBM_URL env provided; using locked dataset: %s", forced)
        return forced
    return find_latest_nbm_opendap_url()


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


def open_nbm_subset(opendap_url: str, basin: Polygon, hours: int) -> Tuple[xr.DataArray, Tuple[float, float, float, float]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SerializationWarning)
        ds = xr.open_dataset(opendap_url, engine="netcdf4", cache=False)
    minx, miny, maxx, maxy = basin.bounds
    pad = 0.05
    lat_slice = coord_slice(ds["lat"].values, miny - pad, maxy + pad)
    lon_slice = coord_slice(ds["lon"].values, minx - pad, maxx + pad)
    qpf = ds["apcpsfc"].sel(lat=lat_slice, lon=lon_slice).isel(time=slice(0, hours)).load()
    ds.close()
    return qpf, (minx, miny, maxx, maxy)


# ----------------------------
# Geometry
# ----------------------------
def get_watershed_basin(cfg: RunConfig, offline: bool = False, streamstats_timeout: int = 30, streamstats_retries: int = 5) -> Polygon:
    basin_dir = cfg.root / cfg.basins_dir / cfg.name
    basin_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = basin_dir / "basin.geojson"
    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
        return gdf.geometry.iloc[0]

    legacy_geo = cfg.root / cfg.basins_dir / "basin.geojson"
    if legacy_geo.exists():
        logging.warning("Using legacy Basins/basin.geojson for %s (migrating to per-basin cache).", cfg.name)
        gdf = gpd.read_file(legacy_geo)
        geom = gdf.geometry.iloc[0]
        gpd.GeoDataFrame(index=[0], geometry=[geom], crs="EPSG:4326").to_file(geojson_path, driver="GeoJSON")
        return geom

    if offline:
        raise RuntimeError(f"No cached basin found for {cfg.name} and offline mode is enabled.\nExpected at: {geojson_path}")

    url = (
        "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson?"
        f"rcode={cfg.state}&xlocation={cfg.longitude}&ylocation={cfg.lattitude}&crs=4326"
        "&includeparameters=false&includeflowtypes=false&includefeatures=true&simplify=true"
    )
    session, timeout = make_retrying_session(total=streamstats_retries, backoff=1.5, timeout=streamstats_timeout)
    logging.info("Requesting StreamStats basin for %s…", cfg.name)
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
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


def load_subbasin_geoms_for_parent(parent_name: str,
                                   root: Path,
                                   basins_dir: Path,
                                   cfg_json: Dict[str, dict]) -> Dict[str, Polygon]:
    """
    Loads (or fetches/caches) each subbasin geometry under the given PARENT.
    """
    out: Dict[str, Polygon] = {}
    pinfo = cfg_json.get(parent_name)
    if not pinfo:
        return out
    subs = pinfo.get("subbasins", {})
    if not isinstance(subs, dict) or not subs:
        return out
    for sub_name, sb in subs.items():
        sub_cfg = RunConfig(
            name=f"{parent_name}/{sub_name}",
            state=str(sb["state"]),
            lattitude=float(sb["latitude"]),
            longitude=float(sb["longitude"]),
            root=root,
            basins_dir=basins_dir,
        )
        try:
            geom = get_watershed_basin(sub_cfg, offline=False)
            out[sub_name] = geom
        except Exception as e:
            logging.warning("Could not load subbasin %s/%s outline: %s", parent_name, sub_name, e)
    return out


# ----------------------------
# Gridding & aggregation
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


def boundary_correct_3d(arr_in_inch: np.ndarray, time_vals) -> np.ndarray:
    arr = np.asarray(arr_in_inch, dtype=float).copy()
    nt = arr.shape[0]
    for t in range(nt):
        hr = int(pd.to_datetime(time_vals[t]).hour)
        if hr in (0, 6, 12, 18):
            t0 = max(0, t - 5)
            arr[t, :, :] = np.maximum(0.0, arr[t, :, :] - np.nansum(arr[t0:t, :, :], axis=0))
    return arr


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
# Plotting (MAP)
# ----------------------------
def plot_cumulative_map(basin: Polygon,
                        lat_f: np.ndarray,
                        lon_f: np.ndarray,
                        dlat_f: float,
                        dlon_f: float,
                        mask: np.ndarray,
                        fine_cum: np.ndarray,
                        title: str,
                        save_basepath: str | None = None,
                        subbasin_outlines: Dict[str, Polygon] | None = None):
    """
    Prettier Matplotlib version:
      - smooth raster with imshow (bilinear)
      - clean frame, subtle grid, nicer labels
      - crisp outlines & readable labels
    """
    import matplotlib.patheffects as pe

    # Mask out of basin
    Z = np.where(mask, fine_cum, np.nan)

    # Map bounds
    minx, miny, maxx, maxy = basin.bounds
    extent = (float(minx), float(maxx), float(miny), float(maxy))

    # Colors (discrete WPC-style bins)
    cmap, norm, levels = wpc_qpf_colormap_extended()

    fig, ax = plt.subplots(figsize=(11, 7))

    # Smooth raster: imshow with bilinear interpolation, proper extent
    im = ax.imshow(
        Z,
        origin="lower",
        extent=extent,
        interpolation="bilinear",   # smooth without looking blurry
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )

    # Subbasin outlines (if provided)
    if subbasin_outlines:
        for name, geom in subbasin_outlines.items():
            gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(
                ax=ax, zorder=3, color="#111", linewidth=1.2
            )
            try:
                rp = geom.representative_point()
                ax.text(float(rp.x), float(rp.y), name.replace("_", " "),
                        fontsize=9, ha="center", va="center",
                        color="#111",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.0),
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                        zorder=4)
            except Exception:
                pass

    # Basin outline on top
    gpd.GeoSeries([basin], crs="EPSG:4326").boundary.plot(
        ax=ax, zorder=4, color="#000", linewidth=1.6
    )

    # Colorbar
    cax = fig.add_axes([0.90, 0.16, 0.02, 0.68])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Cumulative precipitation (in)")
    cb.set_ticks(levels)
    cb.ax.set_yticklabels([f"{lv:g}" for lv in levels])

    # Styling
    ax.set_title(title, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    ax.grid(True, alpha=0.10, linestyle="-", linewidth=0.6)

    # Tighten layout and save
    if save_basepath:
        fig.savefig(f"{save_basepath}.png", dpi=180, bbox_inches="tight")
        fig.savefig(f"{save_basepath}.svg",           bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_map_plotly_then_fallback(
    basin: Polygon,
    lat_f: np.ndarray,
    lon_f: np.ndarray,
    dlat_f: float,
    dlon_f: float,
    mask: np.ndarray,
    fine_cum: np.ndarray,
    title: str,
    save_basepath: str | None = None,
    subbasin_outlines: Dict[str, Polygon] | None = None,
    max_daily_in: float = 6.0,
):
    """
    Plotly heatmap (PNG/SVG via kaleido)
      - DISCRETE NWS/WPC-style colors (hard steps)
      - 1-inch legend tick marks (0..max_daily_in)
      - Sub-basin outlines + labels
    Falls back to Matplotlib on error or if save_basepath is None.
    """
    if save_basepath is None:
        return plot_cumulative_map(
            basin, lat_f, lon_f, dlat_f, dlon_f, mask, fine_cum, title, save_basepath, subbasin_outlines
        )

    # ------------------------------------------------------------------
    def _segments_from_geom(g) -> list[tuple[list[float], list[float]]]:
        from shapely.geometry import Polygon as SPoly, MultiPolygon as SMulti
        segs: list[tuple[list[float], list[float]]] = []
        try:
            if isinstance(g, SPoly):
                x, y = g.exterior.xy
                segs.append((list(x), list(y)))
                for interior in g.interiors:
                    xi, yi = interior.xy
                    segs.append((list(xi), list(yi)))
            elif isinstance(g, SMulti):
                for p in g.geoms:
                    x, y = p.exterior.xy
                    segs.append((list(x), list(y)))
                    for interior in p.interiors:
                        xi, yi = interior.xy
                        segs.append((list(xi), list(yi)))
        except Exception:
            pass
        return segs

    def _discrete_nws_colorscale(max_in: float = 6.0) -> tuple[list[float], list[list]]:
        """
        Return (levels, colorscale) matching NWS/WPC-style discrete bins up to max_in inches.
        Each bin [L_i, L_{i+1}) uses a single color (hard steps in Plotly).
        """
        # Canonical WPC breakpoints (trimmed/capped to max_in)
        levels = [0, 0.01, 0.10, 0.25, 0.50, 1, 1.5, 2, 3, 4, 5, 6]
        colors = [
            "#ffffff",  # 0–0.01  (white / no precip)
            "#e6ffe6",  # 0.01–0.10 very light green
            "#b3ffb3",  # 0.10–0.25 light green
            "#66ff66",  # 0.25–0.50 medium green
            "#ffff66",  # 0.50–1 yellow
            "#ffcc66",  # 1–1.5 light orange
            "#ff9933",  # 1.5–2 orange
            "#ff3300",  # 2–3 red-orange
            "#cc0000",  # 3–4 deep red
            "#990099",  # 4–5 purple
            "#660099",  # 5–6 deep purple
        ]
        # Cap/trim to max_in
        levels = [lv for lv in levels if lv <= max_in]
        colors = colors[: max(1, len(levels) - 1)]
        # Build stepped colorscale (duplicate stops for hard edges)
        lmin, lmax = levels[0], levels[-1]
        rng = max(lmax - lmin, 1e-9)
        colorscale: list[list] = []
        for i in range(len(levels) - 1):
            a = (levels[i] - lmin) / rng
            b = (levels[i + 1] - lmin) / rng
            colorscale.append([a, colors[i]])
            colorscale.append([b, colors[i]])
        colorscale.append([1.0, colors[-1]])
        return levels, colorscale

    # ------------------------------------------------------------------
    try:
        import plotly.graph_objs as go
        from plotly import io as pio
        _ = pio.kaleido  # ensure kaleido is available

        # --- scale setup (discrete) ---
        levels, colorscale = _discrete_nws_colorscale(max_in=max_daily_in)
        lmin, lmax = levels[0], levels[-1]

        # clamp data to top bin so big values don't stretch the scale
        Z = np.where(mask, np.clip(fine_cum, lmin, lmax), np.nan)

        # Convert to lists to avoid array('d') issues
        lon_list = np.asarray(lon_f).tolist()
        lat_list = np.asarray(lat_f).tolist()
        Z_list   = np.asarray(Z).tolist()

        fig = go.Figure()

        # --- Heatmap (discrete steps) ---
        fig.add_trace(go.Heatmap(
            x=lon_list,
            y=lat_list,
            z=Z_list,
            colorscale=colorscale,
            zmin=lmin,
            zmax=lmax,
            zsmooth=False,  # keep bin edges crisp
            colorbar=dict(
                title="in",
                tickvals=[lv for lv in range(int(lmin), int(lmax) + 1)],  # 1-inch ticks
                ticktext=[f"{lv:g}" for lv in range(int(lmin), int(lmax) + 1)],
                len=0.85,
            ),
            hovertemplate="Lon %{x:.3f}, Lat %{y:.3f}<br>%{z:.2f} in<extra></extra>",
        ))

        # --- Sub-basin outlines ---
        if subbasin_outlines:
            for name, geom in subbasin_outlines.items():
                for x_seg, y_seg in _segments_from_geom(geom):
                    fig.add_trace(go.Scatter(
                        x=x_seg, y=y_seg, mode="lines",
                        line=dict(color="#111", width=1.4),
                        hoverinfo="skip", showlegend=False,
                    ))

        # --- Parent basin outline ---
        for x_seg, y_seg in _segments_from_geom(basin):
            fig.add_trace(go.Scatter(
                x=x_seg, y=y_seg, mode="lines",
                line=dict(color="#000", width=2),
                hoverinfo="skip", showlegend=False,
            ))

        # --- Sub-basin annotations (representative points) ---
        annos = []
        if subbasin_outlines:
            for name, geom in subbasin_outlines.items():
                try:
                    rp = geom.representative_point()
                    annos.append(dict(
                        x=float(rp.x), y=float(rp.y),
                        text=name.replace("_", " "),
                        xanchor="center", yanchor="middle",
                        showarrow=False,
                        font=dict(size=12, color="#111"),
                        bgcolor="rgba(255,255,255,0.72)",
                        bordercolor="rgba(0,0,0,0.18)",
                        borderwidth=1,
                        borderpad=2,
                    ))
                except Exception:
                    pass
        if annos:
            fig.update_layout(annotations=annos)

        # --- Layout ---
        fig.update_layout(
            title=title,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=60, r=40, t=70, b=60),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # --- Export ---
        fig.write_image(f"{save_basepath}.png", scale=2)
        fig.write_image(f"{save_basepath}.svg")

        return

    except Exception as e:
        logging.warning("Plotly heatmap export failed (%s); falling back to Matplotlib.", e)
        return plot_cumulative_map(
            basin, lat_f, lon_f, dlat_f, dlon_f, mask, fine_cum, title, save_basepath, subbasin_outlines
        )




# ----------------------------
# Plotting (PLOTLY timeseries)
# ----------------------------
def _to_local_index(ser_utc: pd.Series) -> pd.Series:
    s = ser_utc.copy()
    idx = pd.DatetimeIndex(s.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    s.index = idx.tz_convert("America/Los_Angeles")
    return s


def plotly_ts_precip_multi(parent_hourly_utc: pd.Series,
                           subs_hourly_map_utc: Dict[str, pd.Series] | None,
                           title: str,
                           outfile_html: str,
                           base_label: str = "Parent"):
    """
    Bars: hourly for each series (grouped)
    Lines: cumulative for the same series
    Matching colors per series; legend toggles bar+line together.
    base_label: label used for the 'parent_hourly_utc' series (for subbasin-only plots, pass subbasin name).
    """
    if not _PLOTLY_OK:
        logging.warning("Plotly not available; skipping %s", outfile_html)
        return

    # Collect all series -> local; align to union index
    series_local: Dict[str, pd.Series] = {}
    series_local[base_label] = _to_local_index(parent_hourly_utc)
    for name, ser in (subs_hourly_map_utc or {}).items():
        series_local[name] = _to_local_index(ser)

    all_index = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series_local.values()])))
    for k in list(series_local.keys()):
        series_local[k] = series_local[k].reindex(all_index, fill_value=0.0)

    # Palette (explicit so bars & lines match)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    fig = go.Figure()
    for i, (label, ser) in enumerate(series_local.items()):
        color = palette[i % len(palette)]
        # Bars (no legend entry; grouped by offsetgroup)
        fig.add_bar(
            x=ser.index, y=ser.values,
            name=f"{label} hourly",
            legendgroup=label, showlegend=False,
            offsetgroup=label,
            hovertemplate="%{x|%a %b %d %H:%M}<br>%{y:.2f} in<hr>",
            marker_color=color, opacity=0.75 if i == 0 else 0.55,
        )
        # Cumulative line (legend entry)
        fig.add_trace(go.Scatter(
            x=ser.index, y=ser.cumsum().values,
            mode="lines",
            name=label, legendgroup=label,
            hovertemplate="%{x|%a %b %d %H:%M}<br>Cumulative %{y:.2f} in<hr>",
            line=dict(color=color), marker=dict(color=color),
        ))

    fig.update_layout(
        title=title,
        # xaxis_title="Local time",
        yaxis_title="Precipitation (in)",
        barmode="group",
        legend=dict(
            title="Toggle series",
            groupclick="togglegroup",
            orientation="v",        # vertical legend (stacked)
            yanchor="top",          # align legend top with plot top
            y=1,                    # top of legend
            xanchor="left",         # attach legend’s left edge
            x=1.02,                 # position just right of plot area
            bgcolor="rgba(255,255,255,0.8)",  # optional: white semi-transparent background
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode="x unified",
        margin=dict(l=60, r=140, t=70, b=60),  # add extra right margin for legend
    )

    fig.update_xaxes(showgrid=True, tickformat="%m-%d\n%H:%M")

    # Show/Hide all
    ntr = len(fig.data)
    # fig.update_layout(
    #     updatemenus=[dict(
    #         type="buttons", direction="right", x=0.0, y=1.15, xanchor="left", yanchor="top",
    #         buttons=[
    #             dict(label="Show all", method="update", args=[{"visible":[False]*ntr}]),
    #             dict(label="Hide all", method="update", args=[{"visible":[False]*ntr}]),
    #         ]
    #     )]
    # )

    plotly_plot(fig, filename=outfile_html, auto_open=False, include_plotlyjs="inline")


def plotly_ts_gages(gage_series: List[Tuple[str, pd.Series, str]], title: str, outfile_html: str):
    """
    gage_series: list of (label, series_UTC, units)
    """
    if not _PLOTLY_OK:
        logging.warning("Plotly not available; skipping %s", outfile_html)
        return

    fig = go.Figure()
    unit_set = set()
    any_data = False

    for label, ser_utc, units in gage_series:
        if ser_utc is None or ser_utc.empty or ser_utc.dropna().empty:
            continue
        s_loc = _to_local_index(ser_utc)
        fig.add_trace(go.Scatter(
            x=s_loc.index, y=s_loc.values,
            mode="lines",
            name=label,
            hovertemplate="%{x|%a %b %d %H:%M}<br>%{y:.2f} " + (units or "") + "<hr>",
        ))
        unit_set.add(units or "")
        any_data = True

    if not any_data:
        fig.add_annotation(text="No gage data available", xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        ytitle = ""
    else:
        ytitle = " / ".join(sorted(u for u in unit_set if u))

    fig.update_layout(
        title=title,
        # xaxis_title="Local time",
        yaxis_title=ytitle or "Value",
        hovermode="x unified",
        margin=dict(l=60, r=140, t=70, b=60),  # add space on the right for legend
        legend=dict(
            orientation="v",        # vertical stacking
            yanchor="top",          # align top of legend with top of plot
            y=1,                    # y position (1 = top)
            xanchor="left",         # attach legend’s left edge
            x=1.02,                 # just to the right of plot area
            title="Toggle series",  # optional legend title
            bgcolor="rgba(255,255,255,0.8)",  # semi-transparent background
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
    )

    fig.update_xaxes(showgrid=True, tickformat="%m-%d\n%H:%M")
    plotly_plot(fig, filename=outfile_html, auto_open=False, include_plotlyjs="inline")


# ----------------------------
# Archiving + pruning
# ----------------------------
def archive_parent_basin_grids(
    cfg: RunConfig,
    out_dir: Path,
    lat_f: np.ndarray,
    lon_f: np.ndarray,
    dlat_f: float,
    dlon_f: float,
    times_utc: pd.DatetimeIndex,
    hourly_series_parent: pd.Series,
    fine_cum_parent: np.ndarray,
    nbm_url: str,
    stamp: str,
) -> None:
    if "/" in cfg.name:
        return  # archive parents only
    archive_dir = out_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{cfg.name.replace('/', '__')}_{stamp}.npz"
    fpath = archive_dir / fname

    lat_f32   = np.asarray(lat_f, dtype=np.float32)
    lon_f32   = np.asarray(lon_f, dtype=np.float32)
    dlat_f32  = np.float32(dlat_f)
    dlon_f32  = np.float32(dlon_f)
    fine32    = np.asarray(fine_cum_parent, dtype=np.float32)
    hourly32  = np.asarray(hourly_series_parent.values, dtype=np.float32)
    times64   = np.asarray(pd.DatetimeIndex(times_utc).values)  # np.datetime64[ns]

    meta = {
        "name": cfg.name,
        "state": cfg.state,
        "hours": int(cfg.hours),
        "upscale_factor": int(cfg.upscale_factor),
        "area_weighting": bool(cfg.area_weighting),
        "nbm_url": nbm_url,
        "created_local": stamp,
    }

    np.savez_compressed(
        fpath,
        lat_f=lat_f32,
        lon_f=lon_f32,
        dlat_f=dlat_f32,
        dlon_f=dlon_f32,
        times_utc=times64,
        hourly_series_parent=hourly32,
        fine_cum_parent=fine32,
        meta=json.dumps(meta),
    )
    logging.info("Archived parent grids: %s", fpath)


def prune_old_plots(out_dir: Path, keep: int = 2) -> None:
    """
    Keep only the most-recent `keep` timestamped files per family in `out_dir`.
    Does not touch any '*_latest.*' files.

    Families handled:
      area_precip_*.{png,svg}
      ts_precip_*.html
      gage_*.html
    """
    if not out_dir.exists() or not out_dir.is_dir():
        return

    pat = re.compile(
        r'^(?P<family>'
        r'area_precip|ts_precip|gage'
        r')_(?P<stamp>\d{8}_\d{4})(?:[^.]*)\.(?P<ext>png|svg|html)$'
    )

    buckets: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        if "latest" in p.name:
            continue  # skip rolling pointers
        m = pat.match(p.name)
        if not m:
            continue
        family = m.group("family")
        stamp = m.group("stamp")
        ext = m.group("ext")
        key = (family, ext)
        buckets.setdefault(key, []).append((stamp, p))

    for key, items in buckets.items():
        items.sort(key=lambda sp: sp[0])  # ascending by stamp
        to_delete = items[:-keep] if len(items) > keep else []
        for _, path in to_delete:
            try:
                path.unlink()
                logging.info("Pruned old plot: %s", path.name)
            except Exception as e:
                logging.warning("Could not delete %s: %s", path, e)


# ----------------------------
# Orchestration per basin
# ----------------------------
def run_one(cfg: RunConfig,
            offline: bool = False,
            streamstats_timeout: int = 30,
            streamstats_retries: int = 5):
    # 1) Basin geom
    basin = get_watershed_basin(cfg, offline=offline,
                                streamstats_timeout=streamstats_timeout,
                                streamstats_retries=streamstats_retries)
    logging.info("Basin ready: %s", cfg.name)

    # 2) Forecast
    url = get_locked_nbm_url()
    logging.info("Using forecast: %s", url)
    qpf, bbox = open_nbm_subset(url, basin, cfg.hours)

    # 3) mm -> in, boundary fix
    times_utc = pd.to_datetime(qpf.time.values)
    coarse_in = (qpf.values / 25.4).astype("float64")
    coarse_in_corr = boundary_correct_3d(coarse_in, qpf.time.values)

    # 4) Coordinate order
    lat_src, lon_src, arr3d = ensure_ascending(qpf.lat.values, qpf.lon.values, coarse_in_corr)

    # 5) Fine grid
    lat_f, lon_f, dlat_f, dlon_f = build_fine_grid(lat_src, lon_src, bbox, cfg.upscale_factor)

    # 6) MASK/WEIGHTS for this basin
    mask, weights = basin_mask_and_weights(basin, lat_f, lon_f, cfg.area_weighting)
    hourly_series_this, fine_cum_this = interpolate_and_aggregate(arr3d, lat_src, lon_src, lat_f, lon_f, weights)
    hourly_series_this.index = times_utc  # UTC

    # 7) Output dirs & names
    parent_name = cfg.name.split("/")[0]
    is_parent = ("/" not in cfg.name)

    out_dir_parent = (cfg.root / "docs" / "assets" / parent_name)
    out_dir_this   = (cfg.root / "docs" / "assets" / Path(cfg.name))   # <Parent> or <Parent>/<Sub>
    out_dir_parent.mkdir(parents=True, exist_ok=True)
    out_dir_this.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y%m%d_%H%M")

    # Stamped outputs go to THIS run's folder
    area_base  = str(out_dir_this / f"area_precip_{stamp}")
    ts_path    = str(out_dir_this / f"ts_precip_{stamp}.html")
    gage_path  = str(out_dir_this / f"gage_{stamp}.html")  # parent only used

    # 8) Subbasin outlines + subbasin series (for parent combined plot)
    sub_outlines = {}
    subs_hourly_map: Dict[str, pd.Series] = {}
    if is_parent:
        try:
            cfg_json_all = load_config_file(cfg.root)
            sub_outlines = load_subbasin_geoms_for_parent(cfg.name, cfg.root, cfg.basins_dir, cfg_json_all)
        except Exception as e:
            logging.warning("Subbasin outline load failed for %s: %s", cfg.name, e)
            sub_outlines = {}

        # Build subbasin hourly series using SAME fine grid
        for sub_name, geom in (sub_outlines or {}).items():
            _, w_sub = basin_mask_and_weights(geom, lat_f, lon_f, cfg.area_weighting)
            sub_hourly, _ = interpolate_and_aggregate(arr3d, lat_src, lon_src, lat_f, lon_f, w_sub)
            sub_hourly.index = times_utc  # UTC
            subs_hourly_map[sub_name] = sub_hourly

    # 9) MAP (Matplotlib) — pass outlines only for parent
    title_map = f"{cfg.name} — Basin cumulative precipitation (fine grid x{cfg.upscale_factor})"
    plot_cumulative_map_plotly_then_fallback(
        basin, lat_f, lon_f, dlat_f, dlon_f, mask, fine_cum_this,
        title=title_map,
        save_basepath=area_base,
        subbasin_outlines=sub_outlines if is_parent else None
    )

    # Update area_precip_latest in the run's own folder (parent & subbasin)
    try:
        shutil.copyfile(f"{area_base}.png", out_dir_this / "area_precip_latest.png")
        shutil.copyfile(f"{area_base}.svg", out_dir_this / "area_precip_latest.svg")
    except Exception as e:
        logging.warning("Failed updating area_precip_latest: %s", e)

    # 10) Time-series (Plotly)
    if is_parent:
        # Parent: combined plot (parent + subbasins)
        if _PLOTLY_OK:
            title_ts = f"{cfg.name} — Hourly bars + Cumulative lines (parent + subbasins) — local time"
            plotly_ts_precip_multi(hourly_series_this, subs_hourly_map,
                                   title=title_ts, outfile_html=ts_path, base_label="Parent")
        else:
            with open(ts_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h3>Plotly not available on this runner.</h3></body></html>")

        # Parent latest pointer lives in the parent folder
        try:
            shutil.copyfile(ts_path, out_dir_parent / "ts_precip_latest.html")
        except Exception as e:
            logging.warning("Failed updating parent ts_precip_latest: %s", e)

    else:
        # Subbasin: individual plot (bars + line for this subbasin only)
        if _PLOTLY_OK:
            title_ts = f"{cfg.name} — Hourly bars + Cumulative line — local time"
            plotly_ts_precip_multi(hourly_series_this, {},
                                   title=title_ts, outfile_html=ts_path, base_label=cfg.name.split('/',1)[1])
        else:
            with open(ts_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h3>Plotly not available on this runner.</h3></body></html>")

        # Subbasin latest pointer lives in the SUBFOLDER
        try:
            shutil.copyfile(ts_path, out_dir_this / "ts_precip_latest.html")
        except Exception as e:
            logging.warning("Failed updating subbasin ts_precip_latest: %s", e)

    # 11) Gage plot (Plotly) — parent only
    if is_parent:
        gage_series: List[Tuple[str, pd.Series, str]] = []
        if _GAGE_OK:
            try:
                cfg_json_all = load_config_file(cfg.root)
                pinfo = cfg_json_all.get(parent_name, {})
                gages = pinfo.get("gages", []) if isinstance(pinfo, dict) else []
                for g in gages:
                    try:
                        ser, meta = download_gage(g, hours=cfg.hours)
                        label = g.get("name") or f"{g.get('service','')}-{g.get('site_id','')}"
                        units = meta.get("units", "")
                        gage_series.append((label, ser, units))
                    except Exception as e:
                        logging.warning("Gage fetch failed for %s: %s", g, e)
            except Exception as e:
                logging.warning("Could not load gages for %s: %s", cfg.name, e)

        title_g = f"{cfg.name} — Gage time-series (local time)"
        if gage_series:
            plotly_ts_gages(gage_series, title=title_g, outfile_html=gage_path)
        else:
            with open(gage_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h3>No gage data configured or available.</h3></body></html>")

        # Parent gage latest pointer
        try:
            shutil.copyfile(gage_path, out_dir_parent / "gage_latest.html")
        except Exception as e:
            logging.warning("Failed updating gage_latest: %s", e)

    # 12) Archive parent-basin grids (same stamp) — parent only, saved in parent folder
    if is_parent:
        archive_parent_basin_grids(
            cfg=cfg,
            out_dir=out_dir_parent,
            lat_f=lat_f, lon_f=lon_f, dlat_f=dlat_f, dlon_f=dlon_f,
            times_utc=pd.DatetimeIndex(times_utc),
            hourly_series_parent=hourly_series_this,
            fine_cum_parent=fine_cum_this,
            nbm_url=url,
            stamp=stamp,
        )

    # 13) Prune stamped files (keep current + previous) in THIS run's folder
    prune_old_plots(out_dir_this, keep=2)

    # Optionally also prune in parent folder (helpful if you only run parents sometimes)
    if is_parent:
        prune_old_plots(out_dir_parent, keep=2)


# ----------------------------
# Parallel helpers
# ----------------------------
def _run_one_from_dict(cfg_dict, offline, s_timeout, s_retries):
    try:
        cfg = RunConfig(**cfg_dict)
        run_one(cfg, offline=offline,
                streamstats_timeout=s_timeout,
                streamstats_retries=s_retries)
        return (cfg.name, None)
    except Exception as e:
        return (cfg_dict.get("name", "UNKNOWN"), str(e))


def run_all(root: Path, hours: int = 24, upscale: int = 4, area_weighting: bool = True,
            basins_dir: Path = Path("Basins"),
            offline: bool = False, streamstats_timeout: int = 30, streamstats_retries: int = 5):
    cfg_json = load_config_file(root)
    cfgs = expand_run_configs(cfg_json, hours, upscale, area_weighting, root, basins_dir,
                              only=None, include_subbasins=True)
    for cfg in cfgs:
        try:
            run_one(cfg, offline=offline, streamstats_timeout=streamstats_timeout, streamstats_retries=streamstats_retries)
        except Exception as e:
            logging.exception("Failed for %s: %s", cfg.name, e)


def run_all_parallel(root: Path,
                     hours: int = 24,
                     upscale: int = 4,
                     area_weighting: bool = True,
                     basins_dir: Path = Path("Basins"),
                     offline: bool = False,
                     streamstats_timeout: int = 30,
                     streamstats_retries: int = 5,
                     workers: int | None = None):
    cfg_json = load_config_file(root)
    cfgs = expand_run_configs(cfg_json, hours, upscale, area_weighting, root, basins_dir,
                              only=None, include_subbasins=True)

    # Prefetch geometries (optional, can reduce contention)
    for cfg in cfgs:
        try:
            _ = get_watershed_basin(cfg, offline=offline,
                                    streamstats_timeout=streamstats_timeout,
                                    streamstats_retries=streamstats_retries)
        except Exception as e:
            logging.warning("Prefetch basin failed for %s: %s", cfg.name, e)

    if workers is None or workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 0)

    logging.info("Starting parallel run for %d items (parents+subbasins) with %d workers…", len(cfgs), workers)

    args_list = [(
        cfg.__dict__,
        offline,
        streamstats_timeout,
        streamstats_retries,
    ) for cfg in cfgs]

    any_error = False
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one_from_dict, *args) for args in args_list]
        for fut in as_completed(futs):
            name, err = fut.result()
            if err:
                any_error = True
                logging.error("Basin %s failed: %s", name, err)
            else:
                logging.info("Basin %s finished OK", name)

    if any_error:
        logging.warning("One or more basins/subbasins failed. See logs above.")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NBM basin precipitation → area maps (PNG/SVG) + Plotly timeseries (HTML)."
    )
    parser.add_argument("--root", type=str, default=".",
                        help="Repo root (where Config.json lives). Default: current dir")
    parser.add_argument("--basin", type=str, default=None,
                        help="Run a single parent ('Parent') or sub ('Parent/Sub').")
    parser.add_argument("--all", action="store_true",
        help="Run all parents + subbasins in Config.json")
    parser.add_argument("--hours", type=int, default=24,
                        help="Forecast hours to use (from t=0). Default: 24")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Fine-grid upscale factor (1–4). Default: 4")
    parser.add_argument("--no-area-weighting", action="store_true",
                        help="Disable cos(lat) area weighting")
    parser.add_argument("--basins-dir", type=str, default="Basins",
                        help="Folder where basin caches live. Default: Basins")
    parser.add_argument("--offline", action="store_true",
                        help="Do not call StreamStats; require cached Basins/<name>/basin.geojson.")
    parser.add_argument("--streamstats-timeout", type=int, default=30,
                        help="Per-request timeout for StreamStats (seconds).")
    parser.add_argument("--streamstats-retries", type=int, default=5,
                        help="Number of retries for StreamStats.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for --all (processes). 1 = serial.")
    parser.add_argument("--no-index", action="store_true",
                        help="Skip writing docs/index.html at the end.")
    parser.add_argument("--title", type=str, default="Watershed Forecast Viewer",
                        help="HTML <title> for docs/index.html")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    basins_dir = Path(args.basins_dir)

    if args.all:
        if args.workers and args.workers > 1:
            run_all_parallel(root=root,
                             hours=args.hours,
                             upscale=args.upscale,
                             area_weighting=(not args.no_area_weighting),
                             basins_dir=basins_dir,
                             offline=args.offline,
                             streamstats_timeout=args.streamstats_timeout,
                             streamstats_retries=args.streamstats_retries,
                             workers=args.workers)
        else:
            run_all(root=root,
                    hours=args.hours,
                    upscale=args.upscale,
                    area_weighting=(not args.no_area_weighting),
                    basins_dir=basins_dir,
                    offline=args.offline,
                    streamstats_timeout=args.streamstats_timeout,
                    streamstats_retries=args.streamstats_retries)
    else:
        if not args.basin:
            parser.error("Specify --basin <Parent> or --basin <Parent/Sub>, or use --all")

        cfg_json = load_config_file(root)
        cfgs = expand_run_configs(
            cfg_json,
            hours=args.hours,
            upscale=args.upscale,
            area_weighting=(not args.no_area_weighting),
            root=root,
            basins_dir=basins_dir,
            only=args.basin,
            include_subbasins=True
        )
        for cfg in cfgs:
            run_one(cfg,
                    offline=args.offline,
                    streamstats_timeout=args.streamstats_timeout,
                    streamstats_retries=args.streamstats_retries)

    # Always try to (re)write the index unless explicitly skipped
    if not args.no_index:
        _write_index_after_run(root, title=args.title)



if __name__ == "__main__":
    main()
