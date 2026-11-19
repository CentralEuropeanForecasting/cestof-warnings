#!/usr/bin/env python3
"""Append-only Blitzortung lightning collector with retry-safe snapshot import.

CSV schema remains: latitude,longitude,strike_time,server,mds,mcg,sta
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

DATA_DIR = Path('data')
BLITZ_URL = 'https://map.blitzortung.org/GEOjson/getjson.php?f=s&n=00'
HEADERS = {'Referer': 'https://map.blitzortung.org/', 'User-Agent': 'CESTOF-LightningArchive/1.0'}


def parse_strike_time(value):
    value = str(value).strip()
    # Support timestamps with up to nanosecond precision and optional timezone suffix.
    if '.' in value:
        prefix, rest = value.split('.', 1)
        fraction = ''.join(c for c in rest if c.isdigit())[:6].ljust(6, '0')
        value = f'{prefix}.{fraction}'
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f'Invalid Blitzortung timestamp: {value}')


def fetch_strikes():
    response = requests.get(BLITZ_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list):
        raise ValueError('Lightning provider returned unexpected JSON; refusing update')
    return records


def key_for(row):
    return tuple(row[:4])


def append_snapshot(records):
    DATA_DIR.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    bad = 0
    for strike in records:
        if not isinstance(strike, (list, tuple)) or len(strike) != 7:
            bad += 1
            continue
        lon, lat, strike_time, server, mds, mcg, sta = strike
        try:
            dt = parse_strike_time(strike_time)
            lon_f, lat_f = float(lon), float(lat)
            if not (-180 <= lon_f <= 180 and -90 <= lat_f <= 90):
                raise ValueError('Coordinates out of range')
        except (ValueError, TypeError):
            bad += 1
            continue
        row = [str(lat), str(lon), str(strike_time), str(server), str(mds), str(mcg), str(sta)]
        grouped[DATA_DIR / f'{dt:%Y-%m-%d}.csv'].append(row)

    added = 0
    for path, new_rows in sorted(grouped.items()):
        # Deduplicate against the entire target day's CSV, not just today/yesterday.
        recorded = set()
        previous_count = 0
        if path.exists():
            with path.open(newline='', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 7:
                        recorded.add(key_for(row))
                        previous_count += 1
        fresh = []
        for row in new_rows:
            k = key_for(row)
            if k not in recorded:
                recorded.add(k)
                fresh.append(row)
        if fresh:
            with path.open('a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(fresh)
                f.flush()
                os.fsync(f.fileno())
        added += len(fresh)
        print(f'{path}: existing={previous_count}, appended={len(fresh)}, total={previous_count+len(fresh)}')
    print(f'Incoming={len(records)} invalid={bad} new={added} (CSV rows never deleted)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-out', help='Fetch provider data and save a reusable snapshot; no CSV edits')
    parser.add_argument('--snapshot-in', help='Import captured snapshot into existing daily CSVs')
    args = parser.parse_args()
    if args.snapshot_out and args.snapshot_in:
        parser.error('Choose either --snapshot-in or --snapshot-out')
    if args.snapshot_in:
        with open(args.snapshot_in, encoding='utf-8') as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError('Invalid snapshot format')
    else:
        records = fetch_strikes()
    if args.snapshot_out:
        out = Path(args.snapshot_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(records, f)
        tmp.replace(out)
        print(f'Captured {len(records)} lightning observations at {out}')
    else:
        append_snapshot(records)


if __name__ == '__main__':
    main()
