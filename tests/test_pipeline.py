import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from shapely.geometry import Polygon

import eccodes as ec
import numpy as np
import pandas as pd
import xarray as xr

import build_site
import gageDownload
import nbm_grib as nbm
from basin_map import clipped_overlay

CYCLE = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


class ForecastTests(unittest.TestCase):
    def test_three_day_intervals_cover_each_hour_exactly_once(self):
        steps = nbm.forecast_steps(72)
        self.assertEqual(steps[:24], [(h, 1) for h in range(1, 25)])
        self.assertEqual(steps[24:], [(h, 6) for h in range(30, 73, 6)])
        start = 0
        for end, interval in steps:
            self.assertEqual(end-interval, start)
            start = end
        self.assertEqual(start, 72)
        self.assertEqual(sum(interval for _, interval in steps), 72)
        with self.assertRaises(ValueError):
            nbm.forecast_steps(49)

    def test_long_range_selector_uses_six_hour_total_not_one_hour_sample(self):
        index = ('1:0:d=2026111112:APCP:surface:47-48 hour acc fcst:\n'
                 '2:100:d=2026111112:APCP:surface:42-48 hour acc fcst:\n'
                 '3:250:d=2026111112:TMP:surface:48 hour fcst:\n')
        self.assertEqual(nbm.precipitation_range(index, 48, 6), (100, 249))
        self.assertEqual(nbm.precipitation_range(index, 48, 1), (0, 99))
    def test_hourly_selector_excludes_probabilities_and_six_hour_total(self):
        index = ('1:0:d=2026111112:APCP:surface:5-6 hour acc fcst:prob >0.254:prob fcst 255/255\n'
                 '2:100:d=2026111112:APCP:surface:5-6 hour acc fcst:\n'
                 '3:250:d=2026111112:APCP:surface:0-6 hour acc fcst:\n'
                 '4:400:d=2026111112:TMP:surface:6 hour fcst:\n')
        self.assertEqual(nbm.precipitation_range(index, 6), (100, 249))
        with self.assertRaises(ValueError):
            nbm.precipitation_range(index, 5)

    def test_download_rejects_full_file_response_and_truncation(self):
        index = '1:100:d=2026111112:APCP:surface:0-1 hour acc fcst:\n2:200:d=2026111112:TMP:surface:1 hour fcst:'
        for status, body in [(200, b'full file'), (206, b'GRIBbroken7777')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                response = MagicMock(status_code=status, content=body, headers={'Content-Range': 'bytes 100-199/300'})
                client = MagicMock()
                client.get.side_effect = [MagicMock(text=index), response]
                with patch.object(nbm, 'session') as factory:
                    factory.return_value.__enter__.return_value = client
                    with self.assertRaises((RuntimeError, ValueError)):
                        nbm.download_hour(nbm.AWS, CYCLE, 1, Path(tmp))
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_discovery_never_accepts_old_cycle(self):
        client = MagicMock()
        client.get.side_effect = nbm.requests.ConnectionError('unavailable')
        with patch.object(nbm, 'session') as factory:
            factory.return_value.__enter__.return_value = client
            with self.assertRaises(RuntimeError):
                nbm.discover_cycle(24, CYCLE)
        self.assertEqual(client.get.call_count, 8)
        self.assertFalse(any('20261110/12' in c.args[0] for c in client.get.call_args_list))

    def test_native_lambert_scanning_units_and_accumulation(self):
        gid = ec.codes_grib_new_from_samples('GRIB2')
        try:
            for key, value in {
                'centre': 'kwbc', 'gridType': 'lambert', 'Nx': 4, 'Ny': 3,
                'latitudeOfFirstGridPointInDegrees': 45., 'longitudeOfFirstGridPointInDegrees': 238.,
                'LoVInDegrees': 265., 'LaDInDegrees': 25., 'Latin1InDegrees': 25., 'Latin2InDegrees': 25.,
                'DxInMetres': 2500., 'DyInMetres': 2500., 'shapeOfTheEarth': 6,
                'jScansPositively': 1, 'iScansNegatively': 0, 'alternativeRowScanning': 1,
                'dataDate': 20261111, 'dataTime': 1200, 'discipline': 0, 'parameterCategory': 1,
                'parameterNumber': 8, 'stepType': 'accum', 'startStep': 5, 'endStep': 6,
                'packingType': 'grid_simple', 'bitsPerValue': 24,
            }.items():
                ec.codes_set(gid, key, value)
            # A north/south and east/west ramp detects both flips and transposes.
            expected = np.arange(12, dtype=float).reshape(3, 4)
            raw = expected.copy(); raw[1::2] = raw[1::2, ::-1]
            ec.codes_set_values(gid, raw.ravel())
            # Independent ecCodes coordinates catch an incorrect projection or
            # grid origin in our pyproj-based interpolation.
            lat = ec.codes_get_array(gid, 'latitudes')[6]
            lon = ec.codes_get_array(gid, 'longitudes')[6]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'sample.grib2'
                with path.open('wb') as f:
                    ec.codes_write(gid, f)
                result = nbm.decode_hour(path, CYCLE, 6, np.array([lat]), np.array([lon]))
                self.assertAlmostEqual(result[0, 0], expected[1, 2], places=5)
                with self.assertRaises(ValueError):
                    nbm.decode_hour(path, CYCLE, 5, np.array([lat]), np.array([lon]))
                with self.assertRaises(ValueError):
                    nbm.decode_hour(path, CYCLE.replace(hour=6), 6, np.array([lat]), np.array([lon]))
        finally:
            ec.codes_release(gid)


class SiteTests(unittest.TestCase):
    def test_map_overlay_clips_boundary_without_changing_forecast_units(self):
        geometry = Polygon([(0, 0), (1, 0), (0, 1)])
        axis = np.linspace(-.1, 1.1, 10)
        field = np.full((10, 10), 1.25)
        rgba, maximum = clipped_overlay(geometry, axis, axis, field, pixels=50)
        self.assertAlmostEqual(maximum, 1.25)
        self.assertTrue((rgba[:, :, 3] == 0).any())
        self.assertTrue((rgba[:, :, 3] == 205).any())
        np.testing.assert_array_equal(rgba[rgba[:, :, 3] > 0][:, :3],
                                      np.tile([255, 174, 66], ((rgba[:, :, 3] > 0).sum(), 1)))
        dry, maximum = clipped_overlay(geometry, axis, axis, np.zeros((10, 10)), pixels=50)
        self.assertEqual(maximum, 0)
        self.assertTrue((dry[:, :, 3] == 0).all())
        rectangle = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gradient = np.broadcast_to(.02+.4*axis[:, None], (10, 10))
        north_up, _ = clipped_overlay(rectangle, axis, axis, gradient, pixels=50)
        middle = north_up.shape[1]//2
        # Increasing precipitation to the north belongs at the image's top.
        self.assertLess(north_up[0, middle, 2], north_up[-1, middle, 2])
        with self.assertRaises(ValueError):
            clipped_overlay(geometry, axis, axis, -field, pixels=50)

    def prepare_root(self, root):
        config = {'Test': {'state': 'WA', 'latitude': 45., 'longitude': -122., 'gages': []}}
        (root/'Config.json').write_text(json.dumps(config))
        folder = root/'Basins'/'Test'; folder.mkdir(parents=True)
        polygon = {'type': 'Polygon', 'coordinates': [[[-122.1, 45.], [-122., 45.], [-122., 45.1], [-122.1, 45.1], [-122.1, 45.]]]}
        (folder/'basin.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {}, 'geometry': polygon}]}))

    def test_live_adapter_contract_full_render_and_millimeters_to_inches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare_root(root)
            axis = np.arange(44.9, 45.21, .025)
            longitude = np.arange(-122.2, -121.89, .025)
            data = xr.DataArray(np.full((2, len(axis), len(longitude)), 2.54), dims=('time', 'lat', 'lon'),
                                coords={'time': pd.date_range('2026-11-11T13:00', periods=2, freq='h'), 'lat': axis, 'lon': longitude},
                                attrs={'cycle': CYCLE.isoformat(), 'source': nbm.AWS, 'units': 'mm'})
            with patch.object(build_site, 'load_forecast', return_value=data):
                status = build_site.build(root, root/'output', root/'cache', hours=2)
            frame = pd.read_csv(root/'output/assets/Test/hourly_precip.csv')
            np.testing.assert_allclose(frame.precipitation_inches, [.1, .1])
            self.assertEqual(status['forecast']['hours'], 2)
            html = (root/'output/index.html').read_text()
            self.assertIn('forecast-cycle', html)
            self.assertIn('2h forecast', html)
            self.assertIn('18 * 3600000', html)

    def test_day_controls_totals_and_subbasins_share_one_model_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare_root(root)
            config = json.loads((root/'Config.json').read_text())
            config['Test']['subbasins'] = {'Upper': {'state': 'WA', 'latitude': 45., 'longitude': -122.}}
            (root/'Config.json').write_text(json.dumps(config))
            sub = root/'Basins/Test/Upper'; sub.mkdir()
            (sub/'basin.geojson').write_text((root/'Basins/Test/basin.geojson').read_text())
            steps = nbm.forecast_steps(72)
            axis = np.arange(44.9, 45.21, .025)
            longitude = np.arange(-122.2, -121.89, .025)
            intervals = np.array([duration for _, duration in steps])
            values = np.broadcast_to(intervals[:, None, None]*2.54, (len(steps), len(axis), len(longitude))).copy()
            data = xr.DataArray(values, dims=('time', 'lat', 'lon'),
                                coords={'time': [pd.Timestamp(CYCLE).tz_localize(None)+pd.Timedelta(hours=h) for h, _ in steps],
                                        'lat': axis, 'lon': longitude, 'interval_hours': ('time', intervals)},
                                attrs={'cycle': CYCLE.isoformat(), 'source': nbm.AWS, 'units': 'mm'})
            with patch.object(build_site, 'load_forecast', return_value=data) as fetch:
                status = build_site.build(root, root/'output', root/'cache', days=[1, 2, 3])
            fetch.assert_called_once()
            self.assertEqual(fetch.call_args.args[1], 72)
            self.assertEqual(status['forecast']['available_hours'], [24, 48, 72])
            for hours, rows in [(24, 24), (48, 28), (72, 32)]:
                period = status['forecast']['periods'][str(hours)]
                for name in ['Test', 'Test/Upper']:
                    frame = pd.read_csv(root/'output'/period['assets'][name]['csv'])
                    self.assertEqual(len(frame), rows)
                    self.assertEqual(frame.interval_hours.sum(), hours)
                    self.assertAlmostEqual(frame.precipitation_inches.sum(), hours*.1)
                    self.assertAlmostEqual(period['assets'][name]['total_inches'], hours*.1)
                chart = (root/'output'/period['assets']['Test']['series']).read_text()
                self.assertIn('Upper', chart)
                self.assertIn('interval total', chart)
            self.assertIn('data-hours="72"', (root/'output/index.html').read_text())

    def test_forecast_failure_preserves_site_and_still_fetches_gages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare_root(root)
            config = json.loads((root/'Config.json').read_text())
            config['Test']['gages'] = [{'name': 'Gauge', 'service': 'USGS', 'site_id': '123', 'type': 'FLOW'}]
            (root/'Config.json').write_text(json.dumps(config))
            output = root/'output'; output.mkdir()
            (output/'status.json').write_text('{}')
            (output/'index.html').write_text('last good site')
            with patch.object(build_site, 'fetch_gage', return_value=(None, {'name': 'Gauge', 'state': 'unavailable'})) as fetch:
                with patch.object(build_site, 'load_forecast', side_effect=RuntimeError('upstream unavailable')):
                    with self.assertRaisesRegex(RuntimeError, 'upstream unavailable'):
                        build_site.build(root, output, root/'cache', hours=2)
                fetch.assert_called_once()
            self.assertEqual((output/'index.html').read_text(), 'last good site')
            status = json.loads((root/'failed-build/status.json').read_text())
            self.assertEqual(status['forecast']['state'], 'failed')
            self.assertEqual(len(status['gages']['Test']), 1)

    def test_missing_gage_is_visible_but_not_fatal(self):
        with patch.object(build_site, 'download_gage', side_effect=RuntimeError('API unavailable')):
            data, status = build_site.fetch_gage({'name': 'Gauge', 'service': 'NOAA', 'site_id': 'TEST', 'type': 'STAGE'})
        self.assertIsNone(data)
        self.assertEqual(status['state'], 'unavailable')

    def test_noaa_missing_sentinel_and_empty_payload(self):
        for payload in [{'data': []}, {'primaryName': 'Stage', 'secondaryName': 'Flow', 'data': [{'validTime': '2026-11-11T12:00Z', 'primary': -999, 'secondary': -999}]}]:
            client = MagicMock()
            client.get.return_value.json.return_value = payload
            with patch.object(gageDownload, '_retrying_session', return_value=(client, 20)):
                frame = gageDownload._fetch_noaa_stageflow_df('TEST')
            self.assertTrue(frame.empty or frame[['stage', 'flow']].isna().all().all())

    def test_carson_configuration_uses_noaa_and_converts_kcfs_to_cfs(self):
        root = Path(__file__).resolve().parents[1]
        gauge = json.loads((root/'Config.json').read_text())['Wind_at_Carson']['gages'][0]
        self.assertEqual((gauge['service'], gauge['site_id'], gauge['type']), ('NOAA', 'WCNW1', 'FLOW'))
        payload = {'primaryName': 'Stage', 'primaryUnits': 'ft', 'secondaryName': 'Flow', 'secondaryUnits': 'kcfs',
                   'data': [{'validTime': '2026-11-11T11:15:00Z', 'primary': 3.93, 'secondary': .529}]}
        client = MagicMock(); client.get.return_value.json.return_value = payload
        with patch.object(gageDownload, '_retrying_session', return_value=(client, 20)):
            series, meta = gageDownload.download_gage(gauge, hours=None)
        self.assertAlmostEqual(series.iloc[0], 529)
        self.assertEqual(meta['units'], 'cfs')
        self.assertEqual(meta['service'], 'NOAA')

    def test_usgs_missing_sentinel_is_not_a_measurement(self):
        payload = {'value': {'timeSeries': [{
            'variable': {'noDataValue': -999999.0},
            'values': [{'value': [
                {'dateTime': '2026-11-11T12:00:00Z', 'value': '-999999'},
                {'dateTime': '2026-11-11T12:15:00Z', 'value': '42'},
            ]}],
        }]}}
        client = MagicMock()
        client.get.return_value.json.return_value = payload
        with patch.object(gageDownload, '_retrying_session', return_value=(client, 20)):
            series, _ = gageDownload._download_usgs('TEST', 'FLOW', None)
        self.assertTrue(np.isnan(series.iloc[0]))
        self.assertEqual(series.iloc[1], 42)


if __name__ == '__main__':
    unittest.main()
