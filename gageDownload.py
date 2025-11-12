#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gageDownload.py
---------------
Unified helpers for downloading gage time series from multiple services.

Services:
- USGS (instantaneous values / iv)
- WA_Ecology (TXT tables per station/product)
- NOAA (NWPS stage/flow JSON via api.water.noaa.gov)

All functions return (series_UTC, metadata_dict)
- `series_UTC`: pandas.Series with tz-aware UTC DatetimeIndex
- `metadata_dict`: {service, site_id, name, units, type, ...}

Behavior:
- USGS: parse timestamps (tz-aware from API), convert to UTC.
- WA_Ecology: TXT tables often say "PST" => localize with fixed UTC-8 (no DST),
  else America/Los_Angeles with DST disambiguation.
- NOAA: uses NWPS stage/flow endpoints; we map to FLOW/STAGE and normalize units
  (e.g., kcfs → cfs).
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
# Robust Pacific localization (WA Ecology)
# ----------------------------

def _localize_pacific_with_dst_disambiguation(naive_ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
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
        second_occ = vals.duplicated(keep="first").to_numpy()
        ambiguous_mask = np.ones(len(vals), dtype=bool)
        ambiguous_mask[second_occ] = False
        idx = vals.tz_localize(
            "America/Los_Angeles",
            ambiguous=ambiguous_mask,
            nonexistent="shift_forward",
        )
        return pd.DatetimeIndex(idx)

def _localize_fixed_pst(naive_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return naive_idx.tz_localize("Etc/GMT+8")  # fixed UTC-8 (PST, no DST)


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
    gtype_u = (gtype or "").upper()
    if gtype_u not in _USGS_PARAM_BY_TYPE:
        raise ValueError(f"USGS type must be FLOW or STAGE; got {gtype!r}")

    param_cd, units = _USGS_PARAM_BY_TYPE[gtype_u]

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
    times = [pd.to_datetime(rec["dateTime"], utc=None) for rec in recs]  # tz-aware strings
    values = []
    for rec in recs:
        try:
            values.append(float(rec["value"]))
        except Exception:
            values.append(np.nan)

    idx = _tz_to_utc_index(pd.DatetimeIndex(times))
    ser = pd.Series(values, index=idx).sort_index()

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
    head = "\n".join(text.splitlines()[:40]).upper()
    return " PST" in head or head.strip().endswith("PST.") or "PACIFIC STANDARD TIME" in head

def _parse_wadoe_txt(text: str,
                     expected_type: str) -> Tuple[pd.DataFrame, str, pd.Series, bool]:
    lines = text.splitlines()
    raw = [ln.rstrip("\n") for ln in lines if ln.strip()]

    header_idx = None
    for i, ln in enumerate(raw[:80]):
        if "DATE" in ln and "TIME" in ln and ("Discharge" in ln or "Stage" in ln):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("WA Ecology TXT: could not locate header with DATE/TIME/value columns.")

    sep_idx = None
    for j in range(header_idx + 1, min(header_idx + 6, len(raw))):
        if set(raw[j].strip()) <= set("- "):
            sep_idx = j
            break
    if sep_idx is None:
        sep_idx = header_idx

    header_line = raw[header_idx]
    units = None
    if "Discharge (cfs)" in header_line:
        value_col_name = "Discharge (cfs)"
        units = "cfs"
    elif "Stage (ft)" in header_line:
        value_col_name = "Stage (ft)"
        units = "ft"
    else:
        value_col_name = "VALUE"

    data_rows = []
    for ln in raw[sep_idx + 1:]:
        if ln.startswith("A key to quality codes"):
            break
        if "--" in ln and ln.replace("-", "").strip():
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

    naive_ts = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")
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
    gtype_u = (gtype or "").upper()
    default_product_by_type = {
        "FLOW": ("DSG_FM", "cfs"),
        "STAGE": ("STG_FM", "ft"),
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

    if allow_quality is not None and "QUALITY" in df.columns:
        before = len(df)
        df = df[df["QUALITY"].isin(allow_quality)]
        after = len(df)
        logging.info("WA Ecology: quality filter kept %d/%d rows (codes allowed: %s)", after, before, sorted(allow_quality))

    value_col = "Discharge (cfs)" if gtype_u == "FLOW" else "Stage (ft)"
    if value_col not in df.columns:
        value_col = "VALUE"

    ser = df[value_col].astype(float)
    ser.index = _tz_to_utc_index(ser.index)
    ser = ser.sort_index()
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
# NOAA / NWPS (stage/flow JSON)
# ----------------------------

def _fetch_noaa_stageflow_df(lid: str, product: str = "observed") -> pd.DataFrame:
    """
    Returns tidy DataFrame:
      columns: validTime (UTC tz-aware), stage, stage_units, flow, flow_units
    Uses the 'header + data[]' shape of NWPS stageflow responses.
    """
    url = f"https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow/{product}"
    sess, timeout = _retrying_session()
    logging.info("NOAA NWPS fetch: %s", url)
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    j = r.json()

    p_name  = j.get("primaryName") or "Primary"
    p_units = j.get("primaryUnits") or ""
    s_name  = j.get("secondaryName") or "Secondary"
    s_units = j.get("secondaryUnits") or ""

    rows = []
    for pt in j.get("data", []):
        rows.append({
            "validTime": pd.to_datetime(pt.get("validTime"), utc=True),
            "stage": pt.get("primary") if p_name.lower().startswith("stage") else pt.get("secondary"),
            "flow":  pt.get("secondary") if s_name.lower().startswith("flow") else pt.get("primary"),
        })
    df = pd.DataFrame(rows).sort_values("validTime").reset_index(drop=True)
    df["stage_units"] = p_units if p_name.lower().startswith("stage") else s_units
    df["flow_units"]  = s_units if s_name.lower().startswith("flow")  else p_units
    return df

def _download_noaa(lid: str,
                   gtype: str,
                   hours: Optional[int],
                   name: Optional[str] = None,
                   product: str = "observed") -> Tuple[pd.Series, Dict]:
    """
    Download from NOAA NWPS stageflow for FLOW/STAGE.
    product: "observed" (default) or "forecast"
    """
    gtype_u = (gtype or "").upper()
    if gtype_u not in {"FLOW", "STAGE"}:
        raise ValueError(f"NOAA type must be FLOW or STAGE; got {gtype!r}")

    df = _fetch_noaa_stageflow_df(lid, product=product)
    if df.empty:
        raise ValueError(f"NOAA {lid}: empty stageflow response for {product}")

    # Normalize units (kcfs -> cfs) and select column
    if "flow_units" in df.columns and df["flow_units"].str.lower().eq("kcfs").any():
        df.loc[:, "flow"] = df["flow"] * 1000.0
        df.loc[:, "flow_units"] = "cfs"

    idx = _tz_to_utc_index(pd.DatetimeIndex(df["validTime"]))
    if gtype_u == "FLOW":
        ser = pd.Series(df["flow"].astype(float).to_numpy(), index=idx).sort_index()
        units = (df["flow_units"].dropna().iloc[0] if not df["flow_units"].dropna().empty else "")
    else:
        ser = pd.Series(df["stage"].astype(float).to_numpy(), index=idx).sort_index()
        units = (df["stage_units"].dropna().iloc[0] if not df["stage_units"].dropna().empty else "")

    ser = _window_last_hours(ser, hours)

    meta = {
        "service": "NOAA",
        "site_id": lid,
        "name": name or lid,
        "units": units or ("cfs" if gtype_u == "FLOW" else "ft"),
        "type": gtype_u,
        "product": product,
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
        "service": "USGS" | "WA_Ecology" | "NOAA",
        "site_id": "string",     # USGS station id, Ecology station code, or NOAA LID (e.g., LWWW1)
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

    if service.upper() == "USGS":
        return _download_usgs(site_id, gtype, hours, name=name)

    if service.upper() in {"WA_ECOLOGY", "WA_ECOLOGY".upper(), "WA-ECOLOGY", "WA_Ecology"}:
        return _download_wadoe(site_id, gtype, hours, name=name,
                               product_override=product_override,
                               allow_quality=allow_quality)

    if service.upper() in {"NOAA", "NWPS"}:
        # Default to observed unless caller overrides (e.g., "forecast")
        product = (product_override or "observed").lower()
        return _download_noaa(site_id, gtype, hours, name=name, product=product)

    raise NotImplementedError(f"Service {service!r} not supported yet.")
