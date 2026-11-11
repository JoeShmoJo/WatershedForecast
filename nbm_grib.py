"""Fetch deterministic, one-hour NBM QPF and normalize its native Lambert grid.

The public NOAA bucket requires no credentials. Only the indexed precipitation
message is downloaded, rather than each ~160 MB multi-variable forecast file.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import re

import eccodes as ec
import numpy as np
from pyproj import CRS, Transformer
import requests
from requests.adapters import HTTPAdapter
from scipy.interpolate import RegularGridInterpolator
from urllib3.util.retry import Retry
import xarray as xr

AWS = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"


def session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def forecast_url(base, cycle, hour):
    return f"{base}/blend.{cycle:%Y%m%d/%H}/core/blend.t{cycle:%H}z.core.f{hour:03d}.co.grib2"


def precipitation_range(index, hour):
    """Select precisely the deterministic (hour-1, hour] accumulation."""
    lines = index.splitlines()
    pattern = re.compile(rf"^\d+:\d+:d=\d+:APCP:surface:{hour-1}-{hour} hour acc fcst:$")
    matches = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
    if len(matches) != 1:
        raise ValueError(f"Expected one deterministic one-hour APCP message for f{hour:03d}; found {len(matches)}")
    i = matches[0]
    start = int(lines[i].split(":")[1])
    end = int(lines[i+1].split(":")[1])-1 if i+1 < len(lines) else None
    return start, end


def discover_cycle(hours=24, now=None):
    now = now or datetime.now(timezone.utc)
    candidate = now.replace(hour=(now.hour//6)*6, minute=0, second=0, microsecond=0)
    # Never silently fall back to forecasts more than 18 hours old.
    with session() as s:
        for offset in range(4):
            cycle = candidate - timedelta(hours=6*offset)
            if now-cycle > timedelta(hours=18):
                continue
            for base in (AWS, NOMADS):
                try:
                    for hour in sorted({1, hours}):
                        r = s.get(forecast_url(base, cycle, hour)+".idx", timeout=20)
                        r.raise_for_status()
                        precipitation_range(r.text, hour)
                    logging.info("Selected NBM cycle %s from %s", cycle.isoformat(), base)
                    return cycle, base
                except (requests.RequestException, ValueError) as e:
                    logging.info("NBM cycle %s unavailable from %s: %s", cycle.isoformat(), base, e)
    raise RuntimeError("No complete recent NBM cycle available from NOAA AWS or NOMADS")


def valid_grib(data):
    return len(data) >= 20 and data[:4] == b"GRIB" and data[-4:] == b"7777" and int.from_bytes(data[8:16], "big") == len(data)


def download_hour(base, cycle, hour, cache):
    target = cache / f"nbm_{cycle:%Y%m%d%H}_f{hour:03d}_apcp.grib2"
    if target.exists() and valid_grib(target.read_bytes()):
        return target
    url = forecast_url(base, cycle, hour)
    with session() as s:
        r = s.get(url+".idx", timeout=30)
        r.raise_for_status()
        start, end = precipitation_range(r.text, hour)
        r = s.get(url, headers={"Range": f"bytes={start}-{end if end is not None else ''}"}, timeout=90)
        r.raise_for_status()
        if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise RuntimeError(f"Server did not honor precipitation byte range: {url}")
        if (end is not None and len(r.content) != end-start+1) or not valid_grib(r.content):
            raise ValueError(f"Incomplete GRIB message: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".tmp")
        partial.write_bytes(r.content)
        partial.replace(target)
    return target


def lambert_grid(gid):
    if ec.codes_get(gid, "gridType") != "lambert":
        raise ValueError("Expected an NBM Lambert conformal grid")
    if ec.codes_get(gid, "jPointsAreConsecutive"):
        raise ValueError("Unsupported GRIB scanning order")
    radius = ec.codes_get(gid, "radius")
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={ec.codes_get(gid, 'Latin1InDegrees')} "
        f"+lat_2={ec.codes_get(gid, 'Latin2InDegrees')} "
        f"+lat_0={ec.codes_get(gid, 'LaDInDegrees')} "
        f"+lon_0={ec.codes_get(gid, 'LoVInDegrees')} +R={radius} +units=m"
    )
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = transform.transform(ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                                 ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
    dx = ec.codes_get(gid, "DxInMetres") * (-1 if ec.codes_get(gid, "iScansNegatively") else 1)
    dy = ec.codes_get(gid, "DyInMetres") * (1 if ec.codes_get(gid, "jScansPositively") else -1)
    return transform, x0 + np.arange(ec.codes_get(gid, "Nx"))*dx, y0 + np.arange(ec.codes_get(gid, "Ny"))*dy


def decode_hour(path, cycle, hour, lat, lon):
    with path.open("rb") as f:
        gid = ec.codes_grib_new_from_file(f)
    if gid is None:
        raise ValueError(f"Unreadable GRIB: {path}")
    try:
        actual_cycle = f"{ec.codes_get(gid, 'dataDate'):08d}{ec.codes_get(gid, 'dataTime'):04d}"
        if actual_cycle != cycle.strftime("%Y%m%d%H%M"):
            raise ValueError("GRIB cycle does not match requested cycle")
        if (ec.codes_get(gid, "shortName") != "tp" or ec.codes_get(gid, "stepType") != "accum"
                or ec.codes_get(gid, "startStep") != hour-1 or ec.codes_get(gid, "endStep") != hour
                or ec.codes_get(gid, "stepUnits") != 1):
            raise ValueError(f"Expected one-hour total precipitation for forecast hour {hour}")
        units = ec.codes_get(gid, "units")
        if units not in ("kg m**-2", "kg m-2", "mm"):
            raise ValueError(f"Unexpected precipitation units: {units}")
        transform, x, y = lambert_grid(gid)
        values = ec.codes_get_values(gid).reshape(len(y), len(x))
        if ec.codes_get(gid, "alternativeRowScanning"):
            values[1::2] = values[1::2, ::-1]
        values[values == ec.codes_get(gid, "missingValue")] = np.nan
        xx, yy = transform.transform(*np.meshgrid(lon, lat))
        result = RegularGridInterpolator((y, x), values, bounds_error=True)(np.stack([yy, xx], axis=-1))
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("Missing or negative precipitation in selected region")
        return result
    finally:
        ec.codes_release(gid)


def load_forecast(bounds, hours, cache, workers=4, cycle=None):
    if not 1 <= hours <= 36:
        raise ValueError("Hourly NBM forecasts support --hours 1 through 36")
    if cycle is None:
        cycle, base = discover_cycle(hours)
    else:
        base = AWS
    cache = Path(cache)
    if cache.exists():
        for old in cache.glob("nbm_*_f???_apcp.grib2"):
            match = re.fullmatch(r"nbm_(\d{10})_f\d{3}_apcp.grib2", old.name)
            if match and datetime.strptime(match[1], "%Y%m%d%H").replace(tzinfo=timezone.utc) < cycle-timedelta(days=2):
                old.unlink()
    with ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
        paths = list(pool.map(lambda h: download_hour(base, cycle, h, cache), range(1, hours+1)))
    # Normalize the projected native grid onto a regional regular lat/lon grid;
    # downstream basin masking and cos(latitude) weighting can then be reused.
    x0, y0, x1, y1 = bounds
    lat = np.arange(y0-.1, y1+.125, .025)
    lon = np.arange(x0-.1, x1+.125, .025)
    data = np.stack([decode_hour(p, cycle, h, lat, lon) for h, p in enumerate(paths, 1)])
    times = [np.datetime64((cycle+timedelta(hours=h)).replace(tzinfo=None), "ns") for h in range(1, hours+1)]
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": lon},
                        attrs={"units": "mm", "cycle": cycle.isoformat(), "source": base,
                               "accumulation": "one hour ending at valid time"})
