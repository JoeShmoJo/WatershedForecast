#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gageDownload.py
---------------
Unified helpers for downloading gage time series from multiple services.

Services:
- USGS (instantaneous values / iv)
- WA_Ecology (TXT tables per station/product)

All functions return (series_UTC, metadata_dict)
- `series_UTC`: pandas.Series with tz-aware UTC DatetimeIndex
- `metadata_dict`: {service, site_id, name, units, type, ...}

Behavior:
- USGS: parse timestamps (tz-aware from API), convert to UTC.
- WA_Ecology: TXT tables are often labeled "PST" (UTC-8 year-round).
  If header mentions "PST", we localize using fixed UTC-8 (no DST).
  Otherwise we localize to America/Los_Angeles with robust DST disambiguation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Set

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ----------------------------
# HTTP with retries
# ----------------------------

def _retrying_session(total=4, backoff=1.5, timeout=20) -> Tuple[requests.Session, int]:
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
# Utilities
# ----------------------------

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def _window_last_hours(ser: pd.Series, hours: Optional[int]) -> pd.Series:
    if ser is None or ser.empty or not hours or hours <= 0:
        return ser
    end_utc = _now_utc()
    start_utc = end_utc - pd.Timedelta(hours=hours)
    return ser.loc[(ser.index >= start_utc) & (ser.index <= end_utc)]

def _tz_to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure tz-aware UTC index."""
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


# ----------------------------
# Robust Pacific localization
# ----------------------------

def _localize_pacific_with_dst_disambiguation(naive_ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Localize naive Pacific times to America/Los_Angeles handling DST transitions.

    Strategy:
    - Try fast path with ambiguous='infer'.
    - If AmbiguousTimeError (fall-back hour duplicated), mark first occurrence
      of duplicated wall-times as DST=True, second as DST=False.
    - Handle spring forward with nonexistent='shift_forward'.
    """
    # Convert to Series of naive timestamps if needed
    if isinstance(naive_ts, pd.DatetimeIndex):
        naive = pd.Series(naive_ts)
    else:
        naive = naive_ts.copy()

    try:
        idx = naive.dt.tz_localize(
            "America/Los_Angeles",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        return pd.DatetimeIndex(idx)
    except Exception:
        vals = pd.DatetimeIndex(naive)
        # duplicates -> second occurrence of the same wall time
        second_occ = vals.duplicated(keep="first").to_numpy()
        ambiguous_mask = np.ones(len(vals), dtype=bool)  # True => first occurrence (DST)
        ambiguous_mask[second_occ] = False               # False => second occurrence (standard)
        idx = vals.tz_localize(
            "America/Los_Angeles",
            ambiguous=ambiguous_mask,
            nonexistent="shift_forward",
        )
        return pd.DatetimeIndex(idx)


def _localize_fixed_pst(naive_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Localize naive timestamps to fixed UTC-8 (PST, no DST)."""
    # pandas: 'Etc/GMT+8' means UTC-8 offset (note the sign convention)
    return naive_idx.tz_localize("Etc/GMT+8")


# ----------------------------
# USGS (Instantaneous Values)
# ----------------------------

_USGS_PARAM_BY_TYPE = {
    "FLOW": ("00060", "cfs"),
    "STAGE": ("00065", "ft"),
}

def _download_usgs(site_id: str,
                   gtype: str,
                   hours: Optional[int],
                   name: Optional[str] = None) -> Tuple[pd.Series, Dict]:
    """
    Download USGS IV data for FLOW/STAGE. Returns UTC-indexed series.
    """
    gtype_u = (gtype or "").upper()
    if gtype_u not in _USGS_PARAM_BY_TYPE:
        raise ValueError(f"USGS type must be FLOW or STAGE; got {gtype!r}")

    param_cd, units = _USGS_PARAM_BY_TYPE[gtype_u]

    # Window in UTC (+ small buffer)
    end_utc = _now_utc()
    start_utc = end_utc - pd.Timedelta(hours=(hours or 24) + 2)

    base = "https://waterservices.usgs.gov/nwis/iv/"
    url = (
        f"{base}?format=json"
        f"&sites={site_id}"
        f"&parameterCd={param_cd}"
        f"&startDT={start_utc.strftime('%Y-%m-%dT%H:%MZ')}"
        f"&endDT={end_utc.strftime('%Y-%m-%dT%H:%MZ')}"
        f"&siteStatus=all"
    )

    sess, timeout = _retrying_session()
    logging.info("USGS fetch: %s", url)
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    ts = data.get("value", {}).get("timeSeries", [])
    if not ts:
        raise ValueError(f"USGS site {site_id}: no timeSeries for param {param_cd}")

    vals = ts[0].get("values", [])
    if not vals or not vals[0].get("value"):
        raise ValueError(f"USGS site {site_id}: empty values array for param {param_cd}")

    recs = vals[0]["value"]

    # times include offsets; pandas will parse tz-aware
    times = [pd.to_datetime(rec["dateTime"], utc=None) for rec in recs]
    values = []
    for rec in recs:
        try:
            values.append(float(rec["value"]))
        except Exception:
            values.append(np.nan)

    idx = pd.DatetimeIndex(times)
    idx = _tz_to_utc_index(idx)  # ensure UTC
    ser = pd.Series(values, index=idx).sort_index()

    # Final windowing
    ser = _window_last_hours(ser, hours)

    meta = {
        "service": "USGS",
        "site_id": site_id,
        "name": name or site_id,
        "units": units,
        "parameterCd": param_cd,
        "type": gtype_u,
    }
    return ser, meta


# ----------------------------
# WA Ecology (TXT tables)
# ----------------------------

_WADOE_PRODUCT_BY_TYPE = {
    "FLOW": ("DSG_FM", "cfs"),  # Discharge, 15-min
    "STAGE": ("STG_FM", "ft"),  # Stage, 15-min
}

def _detect_header_says_pst(text: str) -> bool:
    """
    Heuristic: many Ecology files state '... starting at midnight PST.' or
    otherwise mention PST explicitly. If so, treat all times as fixed UTC-8.
    """
    head = "\n".join(text.splitlines()[:40]).upper()
    return " PST" in head or head.strip().endswith("PST.") or "PACIFIC STANDARD TIME" in head


def _parse_wadoe_txt(text: str,
                     expected_type: str) -> Tuple[pd.DataFrame, str, pd.Series, bool]:
    """
    Parse WA Ecology TXT table.

    Returns:
      df (UTC index, columns: VALUE, optional QUALITY),
      units string,
      quality series (or empty),
      used_fixed_pst (bool)
    """
    lines = text.splitlines()
    raw = [ln.rstrip("\n") for ln in lines if ln.strip()]

    # Find header line
    header_idx = None
    for i, ln in enumerate(raw[:80]):
        if "DATE" in ln and "TIME" in ln and ("Discharge" in ln or "Stage" in ln):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("WA Ecology TXT: could not locate header with DATE/TIME/value columns.")

    # dashed separator (optional)
    sep_idx = None
    for j in range(header_idx + 1, min(header_idx + 6, len(raw))):
        if set(raw[j].strip()) <= set("- "):
            sep_idx = j
            break
    if sep_idx is None:
        sep_idx = header_idx

    header_line = raw[header_idx]
    # Identify units/value column
    units = None
    if "Discharge (cfs)" in header_line:
        value_col_name = "Discharge (cfs)"
        units = "cfs"
    elif "Stage (ft)" in header_line:
        value_col_name = "Stage (ft)"
        units = "ft"
    else:
        value_col_name = "VALUE"

    # Collect rows
    data_rows = []
    for ln in raw[sep_idx + 1:]:
        if ln.startswith("A key to quality codes"):
            break
        if "--" in ln and ln.replace("-", "").strip():
            # station title line
            continue
        if set(ln.strip()) <= set("- "):
            continue
        data_rows.append(ln)
    if not data_rows:
        raise ValueError("WA Ecology TXT: no data rows found.")

    records = []
    for ln in data_rows:
        parts = [p for p in ln.split("  ") if p.strip()]
        if len(parts) < 3:
            parts = [p for p in ln.split() if p.strip()]
        if len(parts) < 3:
            continue
        date_str = parts[0]
        time_str = parts[1]
        q = None
        try:
            q = int(parts[-1])
            val_tokens = parts[2:-1]
        except Exception:
            val_tokens = parts[2:]
        val_str = " ".join(val_tokens).replace(",", "")
        try:
            val = float(val_str)
        except Exception:
            val = np.nan
        rec = dict(DATE=date_str, TIME=time_str, VALUE=val)
        if q is not None:
            rec["QUALITY"] = q
        records.append(rec)

    if not records:
        raise ValueError("WA Ecology TXT: parsed zero numeric rows.")

    df = pd.DataFrame.from_records(records)

    # Timestamps (naive)
    naive_ts = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")

    # Decide localization mode
    use_fixed_pst = _detect_header_says_pst(text)
    if use_fixed_pst:
        pacific_idx = _localize_fixed_pst(pd.DatetimeIndex(naive_ts))
        logging.info("WA Ecology: header indicates PST; using fixed UTC-8 (no DST).")
    else:
        pacific_idx = _localize_pacific_with_dst_disambiguation(naive_ts)
        logging.info("WA Ecology: using America/Los_Angeles with DST disambiguation.")

    utc_idx = pacific_idx.tz_convert("UTC")

    df.index = utc_idx
    df = df.drop(columns=[c for c in ("DATE", "TIME") if c in df.columns])
    df = df.sort_index()

    # Units consistency check
    expected_u = (expected_type or "").upper()
    if expected_u == "FLOW" and units not in (None, "cfs"):
        logging.warning("WA Ecology: expected FLOW but detected units=%s", units)
    if expected_u == "STAGE" and units not in (None, "ft"):
        logging.warning("WA Ecology: expected STAGE but detected units=%s", units)

    quality = df["QUALITY"] if "QUALITY" in df.columns else pd.Series(index=df.index, dtype="Int64")
    return df.rename(columns={"VALUE": value_col_name}), (units or ("cfs" if expected_u == "FLOW" else "ft")), quality, use_fixed_pst


def _download_wadoe(site_id: str,
                    gtype: str,
                    hours: Optional[int],
                    name: Optional[str] = None,
                    product_override: Optional[str] = None,
                    allow_quality: Optional[Set[int]] = None) -> Tuple[pd.Series, Dict]:
    """
    Download WA Ecology TXT data. Returns UTC-indexed series.
    """
    gtype_u = (gtype or "").upper()
    default_product_by_type = {
        "FLOW": ("DSG_FM", "cfs"),  # Discharge, 15-min
        "STAGE": ("STG_FM", "ft"),  # Stage, 15-min
    }
    if gtype_u not in default_product_by_type:
        raise ValueError(f"WA_Ecology type must be FLOW or STAGE; got {gtype!r}")

    default_product, default_units = default_product_by_type[gtype_u]
    product = product_override or default_product

    base = "https://apps.ecology.wa.gov/continuousflowandwq"
    url = f"{base}/StationData/Prod/{site_id}/{site_id}_{product}.TXT"

    sess, timeout = _retrying_session()
    logging.info("WA Ecology fetch: %s", url)
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()

    df, units, quality, used_fixed_pst = _parse_wadoe_txt(r.text, expected_type=gtype_u)

    # Filter by QUALITY if requested
    if allow_quality is not None and "QUALITY" in df.columns:
        before = len(df)
        df = df[df["QUALITY"].isin(allow_quality)]
        after = len(df)
        logging.info("WA Ecology: quality filter kept %d/%d rows (codes allowed: %s)", after, before, sorted(allow_quality))

    # Value column back to a unified name
    value_col = "Discharge (cfs)" if gtype_u == "FLOW" else "Stage (ft)"
    if value_col not in df.columns:
        # fallback
        value_col = "VALUE"

    ser = df[value_col].astype(float)
    ser.index = _tz_to_utc_index(ser.index)
    ser = ser.sort_index()

    # Windowing in UTC
    ser = _window_last_hours(ser, hours)

    meta = {
        "service": "WA_Ecology",
        "site_id": site_id,
        "name": name or site_id,
        "units": units or default_units,
        "type": gtype_u,
        "product": product,
        "time_basis": "PST_fixed" if used_fixed_pst else "America/Los_Angeles",
    }
    return ser, meta


# ----------------------------
# Public dispatcher
# ----------------------------

def download_gage(gage: Dict,
                  hours: int = 24,
                  allow_quality: Optional[Set[int]] = None,
                  product_override: Optional[str] = None) -> Tuple[pd.Series, Dict]:
    """
    gage = {
        "service": "USGS" | "WA_Ecology",
        "site_id": "string",
        "type": "FLOW" | "STAGE",
        "state": "WA", "latitude": 45.0, "longitude": -122.0, "name": "Display Name" (optional)
    }
    """
    service = str(gage.get("service", "")).strip()
    site_id = str(gage.get("site_id", "")).strip()
    gtype   = str(gage.get("type", "")).strip().upper()
    name    = gage.get("name")

    if not service or not site_id or gtype not in {"FLOW", "STAGE"}:
        raise ValueError(f"Invalid gage descriptor: {gage}")

    if service == "USGS":
        return _download_usgs(site_id, gtype, hours, name=name)

    if service in {"WA_Ecology", "WA_ECOLOGY", "WA-Ecology"}:
        return _download_wadoe(site_id, gtype, hours, name=name,
                               product_override=product_override,
                               allow_quality=allow_quality)

    raise NotImplementedError(f"Service {service!r} not supported yet.")
