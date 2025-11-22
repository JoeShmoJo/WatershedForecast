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
    :root { --gap: 20px; --card-bg: #fff; --muted: #666; }
    body { font-family: system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #222; background: #fdfdfd; }
    h1 { text-align: center; margin-bottom: 10px; }
    h2 { margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 6px; }
    .basin { margin-bottom: 48px; }
    .plots { display: flex; flex-wrap: wrap; gap: var(--gap); justify-content: space-around; }
    .card { flex: 1 1 420px; background: var(--card-bg); border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); padding: 12px; text-align: center; }
    .card h3 { font-weight: 600; color: #444; margin: 8px 0 12px; font-size: 1.05rem; }
    .card img { max-width: 560px; width: 100%; border: 1px solid #e6e6e6; border-radius: 4px; }
    .iframe-wrap { width: 100%; max-width: 560px; margin: 0 auto; }
    .iframe-wrap iframe { width: 100%; height: 520px; border: 0; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .placeholder { display: grid; place-items: center; height: 520px; border: 2px dashed #cfcfcf; border-radius: 6px; color: var(--muted);
                   background: repeating-linear-gradient(45deg, #fafafa, #fafafa 10px, #f6f6f6 10px, #f6f6f6 20px); }
    .placeholder span { font-size: 0.95rem; }
    .linkrow { margin-top: 8px; }
    .linkrow a { color: #0b5fd7; text-decoration: none; }
    .linkrow a:hover { text-decoration: underline; }
    .topnote { text-align:center; }
    footer { margin-top: 60px; text-align: center; font-size: 0.9em; color: var(--muted); }
    .views { margin-top: 8px; }
    @media (max-width: 600px) { body { margin: 12px; } }
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

def _tile_html(basin_key: str, docs_dir: Path, assets_info: dict) -> str:
    map_png_rel = _rel_from_docs(assets_info["map_png"], docs_dir)
    ts_html_rel  = _rel_from_docs(assets_info["ts_html"], docs_dir)
    gage_html_rel = _rel_from_docs(assets_info["gage_html"], docs_dir)

    card_map  = _img_card("Cumulative Map (next 24h)", map_png_rel)
    card_ts   = _iframe_card("Hourly + Cumulative (next 24h)", ts_html_rel)
    card_gage = _iframe_card("Gage Data (last 24h)", gage_html_rel)

    return f"""
  <div class="basin">
    <h2>{basin_key}</h2>
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

    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
{STYLE}
</head>
<body>
  <h1>Watershed Forecast Plots</h1>
  <p class="topnote">National Blend of Models <a href="https://vlab.noaa.gov/web/mdl/nbm" target="_blank" rel="noopener">NBM Info</a> basin precipitation forecast and USGS streamflow summaries.</p>
  <p class="topnote">Precipitation grids cut to basin. Forecasts updated every ~6 hours.</p>
  <p class="topnote"> TURN YOUR PHONE SIDEWAYS </p>
"""

    body = []
    for basin_key in cfg.keys():  # parents only
        basin_assets = assets_root / basin_key
        info = _detect_assets(basin_assets)
        body.append(_tile_html(basin_key, docs_dir, info))

    ts = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M %Z")
    foot = f"""
  <footer>
    <p>Generated by <code>write_index.py</code> — {ts}</p>
    </p>
  </footer>
  <script data-goatcounter="https://joshmojo.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>



</body>
</html>
"""
    return head + "\n".join(body) + foot


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
