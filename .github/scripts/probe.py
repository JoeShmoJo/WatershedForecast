# .github/scripts/probe.py
# Writes /tmp/probe.json with {"latest_url": "...", "last_modified": "...", "has_new": true/false}
import json
import os
import requests
from datetime import datetime, timedelta, timezone

BASE = "https://nomads.ncep.noaa.gov/dods/blend"
CYCLES = ["18z", "12z", "06z", "00z"]  # newest first

latest = None
for day_offset in range(0, 3):
    date_str = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y%m%d")
    for cyc in CYCLES:
        dds_url = f"{BASE}/blend{date_str}/blend_1hr_{cyc}.dds"
        try:
            r = requests.get(dds_url, timeout=6)
            if r.status_code == 200 and "Dataset" in r.text[:200]:
                latest = f"{BASE}/blend{date_str}/blend_1hr_{cyc}"
                break
        except requests.RequestException:
            pass
    if latest:
        break

# Lightweight freshness header for the dataset (same URL may update later)
last_modified = ""
if latest:
    try:
        head = requests.head(latest + ".das", timeout=6)
        last_modified = head.headers.get("Last-Modified", "")
    except requests.RequestException:
        pass

marker_path = ".github/last_forecast_marker.txt"
prev_url, prev_lm = "", ""
if os.path.exists(marker_path):
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            if lines:
                prev_url = lines[0]
            if len(lines) > 1:
                prev_lm = lines[1]
    except Exception:
        pass

has_new = bool(latest) and (latest != prev_url or last_modified != prev_lm)

with open("/tmp/probe.json", "w") as f:
    json.dump(
        {
            "latest_url": latest or "",
            "last_modified": last_modified,
            "has_new": has_new,
        },
        f,
    )
