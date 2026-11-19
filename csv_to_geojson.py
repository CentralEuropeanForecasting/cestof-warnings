#!/usr/bin/env python3
"""Create today's European GeoJSON from append-only daily strike CSV.

GeoJSON is a derived snapshot, not the permanent archive. Never edits CSV files.
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA = Path('data')
today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
source = DATA / f'{today}.csv'
daily = DATA / f'{today}_europe.geojson'
latest = DATA / 'latest_europe.geojson'

features = []
invalid = 0
if source.is_file():
    with source.open(newline='', encoding='utf-8') as infile:
        for row in csv.reader(infile):
            if len(row) < 7:
                invalid += 1
                continue
            lat_s, lon_s, stamp, server, mds, mcg, sta = row[:7]
            try:
                lat, lon = float(lat_s), float(lon_s)
            except ValueError:
                invalid += 1
                continue
            if not (34 <= lat <= 72 and -25 <= lon <= 45):
                continue
            try:
                hour = int(str(stamp).replace('T', ' ')[11:13])
                if not 0 <= hour <= 23:
                    hour = None
            except (ValueError, TypeError):
                hour = None
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'strike_time': stamp, 'hour': hour,
                    'server': int(server) if server.isdigit() else server,
                    'mds': int(mds) if mds.isdigit() else mds,
                    'mcg': int(mcg) if mcg.isdigit() else mcg,
                    'sta': int(sta) if sta.isdigit() else sta,
                },
            })
else:
    print(f'No UTC-day CSV yet: {source}; producing empty daily GeoJSON (no CSV deleted)')

DATA.mkdir(exist_ok=True)
payload = json.dumps({'type':'FeatureCollection', 'features':features}, separators=(',', ':'))
for output in (daily, latest):
    temp = output.with_name(output.name + '.tmp')
    temp.write_text(payload, encoding='utf-8')
    os.replace(temp, output)
print(f'UTC day {today}: {len(features)} European strikes, {invalid} invalid rows; CSV unchanged')
