#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate docs/index.html from docs/assets/, driven by top-level basins in Config.json.

Tiles per PARENT basin:
  1) area_precip_latest.png                 (image, clickable)
  2) ts_precip_latest.html                  (iframe + link)
  3) gage_latest.html                       (iframe + link)

- Uses paths relative to docs/ so it works via file:// and on GitHub Pages.
- If a "latest" file is missing, falls back to the most recent stamped file.
- Only shows PARENT basins (top-level keys in Config.json).
"""

from __future__ import annotations
import argparse
import json
import re
from html import escape
from pathlib import Path
import pandas as pd

# ---- current, 3-tile outputs ----
MAP_PNG = "area_precip_latest.png"
TS_HTML = "ts_precip_latest.html"
GAGE_HTML = "gage_latest.html"

# Fallback patterns if *_latest.* is missing
PATTERNS = {
    "map_png":  re.compile(r"^area_precip_(\d{8}_\d{4})\.png$"),
    "ts_html":  re.compile(r"^ts_precip_(\d{8}_\d{4})\.html$"),
    "gage_html":re.compile(r"^gage_(\d{8}_\d{4})\.html$"),
}

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _rel_from_docs(p: Path | None, docs_dir: Path) -> str | None:
    """Return a POSIX-style path relative to docs/ (or None)."""
    if p is None:
        return None
    try:
        return p.relative_to(docs_dir).as_posix()
    except Exception:
        # Fallback: try to find 'docs' segment and slice after it
        s = p.as_posix()
        marker = "/docs/"
        i = s.lower().find(marker)
        if i != -1:
            return s[i + len(marker):]
        return None

def _pick_latest_by_pattern(folder: Path, key: str) -> Path | None:
    """
    If a 'latest' file is missing, find the newest stamped file
    matching PATTERNS[key] and return its Path.
    """
    pat = PATTERNS[key]
    best: tuple[str, Path] | None = None
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        stamp = m.group(1)  # YYYYMMDD_HHMM
        if (best is None) or (stamp > best[0]):
            best = (stamp, p)
    return best[1] if best else None

def _detect_assets(assets_dir: Path) -> dict:
    """Return presence of key assets in this basin folder (with fallbacks)."""
    info = {"map_png": None, "ts_html": None, "gage_html": None}

    # Prefer *_latest.*; if missing, fallback to newest stamped
    mp = assets_dir / MAP_PNG
    if _exists(mp):
        info["map_png"] = mp
    else:
        info["map_png"] = _pick_latest_by_pattern(assets_dir, "map_png")

    th = assets_dir / TS_HTML
    if _exists(th):
        info["ts_html"] = th
    else:
        info["ts_html"] = _pick_latest_by_pattern(assets_dir, "ts_html")

    gh = assets_dir / GAGE_HTML
    if _exists(gh):
        info["gage_html"] = gh
    else:
        info["gage_html"] = _pick_latest_by_pattern(assets_dir, "gage_html")

    return info

STYLE = r"""
  <style>
    :root { --gap: 18px; --card-bg: #fff; --muted: #607181; --ink: #20394c; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0 auto; max-width: 1640px; padding: 30px 24px; color: var(--ink); background: #f3f7f9; }
    h1 { text-align: center; margin: 6px 0 12px; font-size: clamp(1.8rem, 3vw, 2.4rem); letter-spacing: -.035em; }
    h1 + p { text-align: center; color: var(--muted); font-size: .9rem; }
    h2 { margin: 0 0 12px; font-size: 1.55rem; letter-spacing: -.02em; }
    .basin { margin: 34px 0 44px; }
    .basin > p { color: var(--muted); font-size: .8rem; }
    .plots { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--gap); }
    .card { min-width: 0; background: var(--card-bg); border: 1px solid #dce5eb; border-radius: 12px; box-shadow: 0 3px 12px #17364a06; padding: 15px 12px; text-align: center; }
    .card h3 { font-weight: 600; margin: 2px 0 16px; font-size: .95rem; }
    .card img { max-width: 560px; width: 100%; border-radius: 6px; }
    .iframe-wrap { width: 100%; margin: 0 auto; }
    .iframe-wrap iframe { width: 100%; height: 440px; border: 0; border-radius: 6px; }
    .placeholder { display: grid; place-items: center; height: 440px; border: 1px dashed #cfcfcf; border-radius: 6px; color: var(--muted); background: #f8fafb; }
    .linkrow { margin-top: 10px; font-size: .85rem; }
    a { color: #087ca9; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .topnote { text-align: center; color: var(--muted); font-size: .85rem; margin: 8px 0; }
    #forecast-duration { justify-content: center; }
    #forecast-duration .duration-label { flex-basis: 100%; text-align: center; font-size: .85rem; color: var(--muted); }
    #forecast-duration button { min-width: 100px; }
    button:focus-visible, a:focus-visible { outline: 3px solid #e7b34e; outline-offset: 3px; }
    .basin-totals { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 16px; }
    .basin-totals span { padding: 7px 12px; border: 1px solid #d6e5ec; border-radius: 7px; background: #fff; font-size: .85rem; }
    .basin-totals span:first-child { color: #086a93; background: #eaf4f9; border-color: #c9dfeb; font-weight: 650; }
    footer { margin-top: 50px; text-align: center; font-size: .8rem; color: var(--muted); }
    @media (max-width: 1100px) { .plots { grid-template-columns: repeat(2, minmax(0, 1fr)); } .plots > .card:first-child { grid-column: 1 / -1; } .card img { max-width: 550px; } }
    @media (max-width: 640px) { body { padding: 18px 12px; } .plots { grid-template-columns: 1fr; } .plots > .card:first-child { grid-column: auto; } .iframe-wrap iframe { height: 400px; } .basin { margin-top: 28px; } #forecast-duration button { min-width: 0; flex: 1; padding: 9px 12px; } }
  </style>
"""



def _img_card(title: str, png_rel: str | None) -> str:
    if png_rel:
        return f"""
      <div class="card">
        <h3>{title}</h3>
        <a href="{png_rel}" target="_blank" rel="noopener">
          <img src="{png_rel}" alt="{title}" />
        </a>
        <div class="linkrow"><a href="{png_rel}" target="_blank" rel="noopener">Open full size</a></div>
      </div>"""
    else:
        return f"""
      <div class="card">
        <h3>{title}</h3>
        <div class="placeholder"><span>No data yet</span></div>
      </div>"""

def _iframe_card(title: str, html_rel: str | None, link_label: str = "Open interactive") -> str:
    if html_rel:
        return f"""
      <div class="card">
        <h3>{title}</h3>
        <div class="iframe-wrap">
          <iframe src="{html_rel}" loading="lazy"></iframe>
        </div>
        <div class="linkrow"><a href="{html_rel}" target="_blank" rel="noopener">{link_label}</a></div>
      </div>"""
    else:
        return f"""
      <div class="card">
        <h3>{title}</h3>
        <div class="placeholder"><span>No data yet</span></div>
      </div>"""

def _tile_html(basin_key: str, docs_dir: Path, assets_info: dict, status: dict | None = None) -> str:
    map_png_rel = _rel_from_docs(assets_info["map_png"], docs_dir)
    ts_html_rel  = _rel_from_docs(assets_info["ts_html"], docs_dir)
    gage_html_rel = _rel_from_docs(assets_info["gage_html"], docs_dir)

    hours = (status or {}).get("forecast", {}).get("hours", 24)
    interactive_map = (status or {}).get("forecast", {}).get("periods", {}).get(str(hours), {}).get("assets", {}).get(basin_key, {}).get("interactive_map")
    if interactive_map:
        card_map = f'''<div class="card"><h3>Cumulative Map ({hours}h forecast)</h3>
        <div class="iframe-wrap"><iframe data-precip-map src="{escape(interactive_map, quote=True)}" loading="lazy" title="{escape(basin_key.replace('_', ' '), quote=True)} precipitation map"></iframe></div>
        <div class="linkrow"><a data-map-full href="{escape(interactive_map, quote=True)}" target="_blank" rel="noopener">Open interactive map</a> · <a data-map-png href="{escape(map_png_rel or '', quote=True)}" target="_blank" rel="noopener">PNG</a></div></div>'''
    else:
        card_map = _img_card(f"Cumulative Map ({hours}h forecast)", map_png_rel)
    card_ts   = _iframe_card(f"Precipitation + Cumulative ({hours}h forecast)", ts_html_rel)
    card_gage = _iframe_card("Gage Data (last 24h)", gage_html_rel)
    observations = []
    for gage in (status or {}).get("gages", {}).get(basin_key, []):
        label = escape(f"{gage['name']} ({gage['service'].replace('_', ' ')})")
        if gage.get("latest"):
            observations.append(f"{label}: <time data-observation-time='{escape(gage['latest'])}'>{escape(gage['latest'])}</time>")
        else:
            observations.append(f"{label}: unavailable")
    observation_html = "<p>Observations — " + "; ".join(observations) + "</p>" if observations else ""

    return f"""
  <div class="basin" data-basin="{escape(basin_key, quote=True)}">
    <h2>{escape(basin_key.replace("_", " "))}</h2>
    <div class="basin-totals" aria-live="polite"></div>
    {observation_html}
    <div class="plots">
      {card_map}
      {card_ts}
      {card_gage}
    </div>
  </div>
"""

def build_index_html(root: Path, title: str) -> str:
    docs_dir = root / "docs"
    assets_root = docs_dir / "assets"
    cfg = json.loads((root / "Config.json").read_text(encoding="utf-8"))
    status_path = docs_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    forecast = status.get("forecast", {})
    periods = forecast.get("periods", {})
    duration_controls = ""
    if len(periods) > 1:
        buttons = "".join(f'<button type="button" data-hours="{h}" aria-pressed="{str(int(h)==forecast["hours"]).lower()}">{int(h)//24} day{"s" if int(h)>24 else ""}</button>' for h in periods)
        duration_controls = f'''<div id="forecast-duration" role="group" aria-label="Forecast duration"><span class="duration-label">Forecast duration</span>{buttons}</div>
        <p class="topnote">Totals start at the same model initialization. Select more days to extend the forecast.</p>
        <p class="topnote" id="interval-note">{escape(periods[str(forecast['hours'])]['interval_note'])}</p>
        <style>#forecast-duration{{display:flex;flex-wrap:wrap;gap:8px;border:0;padding:0;margin:18px 0 8px}}#forecast-duration .duration-label{{font-weight:600;margin-bottom:8px}}#forecast-duration button{{font:inherit;padding:9px 18px;border:1px solid #aebfc9;border-radius:6px;background:white;cursor:pointer}}#forecast-duration button[aria-pressed=true]{{background:#087fb0;color:white;border-color:#087fb0}}</style>'''
    if forecast.get("state") == "ok":
        freshness = f"""<p>Forecast initialized: <time id="forecast-cycle" datetime="{escape(forecast['cycle'])}">{escape(forecast['cycle'])}</time>.
        Forecast period: <time id="forecast-start">{escape(forecast['valid_start'])}</time> through <time id="forecast-end">{escape(forecast['valid_end'])}</time>.</p>
        <p id="freshness-warning" role="status" hidden style="color:#a32900;font-weight:bold">Forecast data is stale. The latest update has not arrived.</p>"""
    else:
        freshness = '<p role="status">Forecast freshness has not been verified.</p>'

    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
{STYLE}
</head>
<body>
  <h1>Watershed Forecasts</h1>
  {freshness}
  {duration_controls}
  <p class="topnote">National Blend of Models <a href="https://vlab.noaa.gov/web/mdl/nbm" target="_blank" rel="noopener">NBM Info</a> basin precipitation forecast and USGS streamflow summaries.</p>
  <p class="topnote">Precipitation grids cut to basin. Forecasts updated every ~6 hours.</p>

"""

    body = []
    for basin_key in cfg.keys():  # parents only
        basin_assets = assets_root / basin_key
        info = _detect_assets(basin_assets)
        body.append(_tile_html(basin_key, docs_dir, info, status))

    ts = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M %Z")
    foot = f"""
  <footer>
    <p>Generated by <code>write_index.py</code> — {ts}</p>
  </footer>
  <script data-goatcounter="https://joshmojo.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>



</body>
</html>
"""
    freshness_script = """<script>
function updateFreshness() {
  const cycle = document.getElementById('forecast-cycle');
  if (cycle) {
    document.getElementById('freshness-warning').hidden = Date.now() - Date.parse(cycle.dateTime) <= 18 * 3600000;
    cycle.textContent = new Date(cycle.dateTime).toLocaleString(undefined, {month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
  }
  document.querySelectorAll('[data-observation-time]').forEach(el => {
    const stale = Date.now() - Date.parse(el.dataset.observationTime) > 3 * 3600000;
    el.style.color = stale ? '#a32900' : '';
    el.title = stale ? 'Observation is more than 3 hours old' : 'Recent observation';
    el.textContent = new Date(el.dataset.observationTime).toLocaleString(undefined, {timeZoneName:'short'}) + (stale ? ' (stale)' : '');
  });
}
updateFreshness(); setInterval(updateFreshness, 60000);
</script>"""
    duration_script = '''<script id="forecast-periods" type="application/json">'''+json.dumps(periods).replace('<', '\\u003c')+'''</script><script>
const forecastPeriods = JSON.parse(document.getElementById('forecast-periods').textContent);
function selectForecast(hours) {
  const period = forecastPeriods[String(hours)];
  if (!period) return;
  const format = value => new Date(value).toLocaleString(undefined, {month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
  document.getElementById('forecast-start').textContent = format(period.valid_start);
  document.getElementById('forecast-end').textContent = format(period.valid_end);
  const note = document.getElementById('interval-note');
  if (note) note.textContent = period.interval_note;
  document.querySelectorAll('[data-hours]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.hours) === Number(hours))));
  document.querySelectorAll('[data-basin]').forEach(basin => {
    const assets = period.assets[basin.dataset.basin];
    if (!assets) return;
    const totals = basin.querySelector('.basin-totals');
    if (totals) {
      const names = [basin.dataset.basin, ...Object.keys(period.assets).filter(name => name.startsWith(basin.dataset.basin + '/'))];
      totals.replaceChildren(...names.map(name => {
        const item = document.createElement('span');
        const label = name === basin.dataset.basin ? 'Basin average' : name.split('/').slice(1).join(' / ').replaceAll('_', ' ');
        item.textContent = `${label}: ${period.assets[name].total_inches.toFixed(2)} in`;
        return item;
      }));
    }
    const cards = basin.querySelectorAll('.plots > .card');
    cards[0].querySelector('h3').textContent = `Cumulative Map (${hours}h forecast)`;
    const image = cards[0].querySelector('img');
    if (image) { image.src = assets.map; image.alt = `Cumulative precipitation over ${hours} hours`; }
    const mapFrame = cards[0].querySelector('iframe[data-precip-map]');
    if (mapFrame && mapFrame.getAttribute('src') !== assets.interactive_map) mapFrame.src = assets.interactive_map;
    cards[0].querySelectorAll('a[data-map-full]').forEach(link => link.href = assets.interactive_map);
    cards[0].querySelectorAll('a[data-map-png]').forEach(link => link.href = assets.map);
    if (!mapFrame) cards[0].querySelectorAll('a').forEach(link => link.href = assets.map);
    cards[1].querySelector('h3').textContent = `Precipitation + Cumulative (${hours}h forecast)`;
    const frame = cards[1].querySelector('iframe');
    if (frame && frame.getAttribute('src') !== assets.series) frame.src = assets.series;
    cards[1].querySelectorAll('a').forEach(link => link.href = assets.series);
  });
}
document.querySelectorAll('[data-hours]').forEach(button => button.addEventListener('click', () => selectForecast(button.dataset.hours)));
selectForecast(Object.keys(forecastPeriods)[0]);
</script>'''
    return head + "\n".join(body) + freshness_script + duration_script + foot


def main():
    ap = argparse.ArgumentParser(description="Write docs/index.html (relative paths; three tiles per parent basin).")
    ap.add_argument("--root", default=".", help="Repository root.")
    ap.add_argument("--title", default="Watershed Forecast Viewer", help="HTML <title> text.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    html = build_index_html(root, args.title)

    out = root / "docs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
