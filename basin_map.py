"""Interactive topographic display of the existing NBM basin precipitation.

This module only renders already-calculated totals. It does not select a model,
fetch a second forecast, or change the basin-average precipitation calculation.
"""
from html import escape
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely import contains_xy
from shapely.geometry import mapping

LEVELS = [0, .01, .10, .25, .50, 1, 2, 3, 4, 5, 6, 8, 10]
COLORS = ['#ffffff', '#c8facc', '#7be37c', '#c7f266', '#ffe66b', '#ffae42',
          '#ff6d3a', '#e53228', '#b3175b', '#6a1b9a', '#4a148c', '#311b92', '#1a237e']


def clipped_overlay(geometry, source_lat, source_lon, cumulative_inches, pixels=240):
    """Resample onto a north-up Web Mercator display raster and mask the basin."""
    forward = Transformer.from_crs(4326, 3857, always_xy=True)
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)
    west, south, east, north = geometry.bounds
    x0, y0 = forward.transform(west, south); x1, y1 = forward.transform(east, north)
    scale = pixels/max(x1-x0, y1-y0)
    width, height = max(16, round((x1-x0)*scale)), max(16, round((y1-y0)*scale))
    xx = x0+(np.arange(width)+.5)*(x1-x0)/width
    yy = y0+(np.arange(height)+.5)*(y1-y0)/height
    lon, lat = inverse.transform(*np.meshgrid(xx, yy))
    inside = contains_xy(geometry, lon, lat)
    values = RegularGridInterpolator((source_lat, source_lon), cumulative_inches, bounds_error=True)(np.stack([lat, lon], axis=-1))
    if not inside.any() or not np.isfinite(values[inside]).all() or (values[inside] < 0).any():
        raise ValueError('Missing or invalid precipitation in interactive basin map')
    palette = np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)] for color in COLORS], dtype=np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = palette[np.clip(np.searchsorted(LEVELS, values, side='right')-1, 0, len(COLORS)-1)]
    # Dry cells remain transparent so terrain is visible. They are not missing.
    rgba[:, :, 3] = (inside & (values >= .01)).astype(np.uint8)*205
    return np.flipud(rgba), float(values[inside].max())


def write_basin_map(folder, site, name, geometry, outlines, totals, source_lat, source_lon, cumulative_inches, period):
    rgba, maximum = clipped_overlay(geometry, source_lat, source_lon, cumulative_inches)
    Image.fromarray(rgba).save(folder/'precip_overlay.png')
    features = []
    for label, shape in [(name, geometry), *outlines.items()]:
        features.append({'type': 'Feature', 'geometry': mapping(shape), 'properties': {
            'name': label.replace('_', ' '), 'total_inches': float(totals[label]), 'parent': label == name}})
    payload = {'name': name.replace('_', ' '), 'hours': period['hours'],
               'start': period['valid_start'], 'end': period['valid_end'],
               'bounds': [[geometry.bounds[1], geometry.bounds[0]], [geometry.bounds[3], geometry.bounds[2]]],
               'features': features, 'maximum_inches': maximum, 'levels': LEVELS, 'colors': COLORS}
    prefix = Path(os.path.relpath(site/'map_assets', folder)).as_posix()
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    title = escape(name.replace('_', ' '))
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} precipitation map</title>
<link rel="stylesheet" href="{prefix}/vendor/leaflet.css"><link rel="stylesheet" href="{prefix}/map.css">
<script>window.BASIN_MAP={encoded};</script><script defer src="{prefix}/vendor/leaflet.js"></script><script defer src="{prefix}/map.js"></script>
</head><body><header><strong id="map-title"></strong><span id="map-period"></span></header>
<div class="map-wrap"><div id="map" aria-label="Interactive basin precipitation map"></div><div class="legend"><strong>Precipitation · inches</strong><div id="legend"></div></div><div id="dry" hidden>Less than 0.01 in forecast throughout this basin.</div></div>
<div id="map-error" role="status" hidden>Interactive map unavailable. Use the PNG download on the forecast page.</div>
</body></html>'''
    (folder/'map.html').write_text(html)
