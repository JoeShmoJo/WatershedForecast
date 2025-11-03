# .github/scripts/probe.py
# Writes /tmp/probe.json with {"latest_url": "...", "has_new": true/false}
import json, os, requests
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

last_file = ".github/last_forecast_url.txt"
prev = ""
if os.path.exists(last_file):
    with open(last_file, "r", encoding="utf-8") as f:
        prev = f.read().strip()

has_new = bool(latest and latest != prev)
with open("/tmp/probe.json", "w") as f:
    json.dump({"latest_url": latest or "", "has_new": has_new}, f)
