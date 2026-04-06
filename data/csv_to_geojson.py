#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from datetime import datetime
from pathlib import Path
import sys

# -----------------------------
# DEBUG
# -----------------------------
print("RUNNING:", __file__)

# -----------------------------
# INPUT LOGIC
# -----------------------------
if len(sys.argv) > 1:
    # Manual mode
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"❌ CSV not found: {input_path}")
        raise SystemExit(1)
else:
    # Auto mode → try today's file first
    today_name = datetime.now().strftime("%Y-%m-%d") + ".csv"
    today_path = Path(today_name)

    if today_path.exists():
        input_path = today_path
        print("Using TODAY file")
    else:
        # fallback → newest file
        csv_files = sorted(Path(".").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files:
            print("❌ No CSV files found")
            raise SystemExit(1)
        input_path = csv_files[0]
        print("Using NEWEST file")

print("Using CSV:", input_path)

# -----------------------------
# OUTPUT DATE
# -----------------------------
stem = input_path.stem
try:
    file_date = datetime.strptime(stem[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
except:
    file_date = datetime.now().strftime("%Y-%m-%d")

OUTPUT_FILE = f"{file_date}_europe.geojson"
LATEST_FILE = "data/latest_europe.geojson"

Path("data").mkdir(exist_ok=True)

# -----------------------------
# EUROPE BOUNDS
# -----------------------------
MIN_LAT = 34.0
MAX_LAT = 72.0
MIN_LON = -25.0
MAX_LON = 45.0

def is_in_europe(lat, lon):
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON

features = []
kept = 0
skipped = 0

# -----------------------------
# LOAD CSV
# -----------------------------
with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 7:
            skipped += 1
            continue

        lat, lon, strike_time, server, mds, mcg, sta = row[:7]

        try:
            lat = float(lat)
            lon = float(lon)
        except:
            skipped += 1
            continue

        if not is_in_europe(lat, lon):
            skipped += 1
            continue

        hour = None
        try:
            hour = datetime.fromisoformat(strike_time).hour
        except:
            pass

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "strike_time": strike_time,
                "hour": hour,
                "server": int(server) if str(server).isdigit() else server,
                "mds": int(mds) if str(mds).isdigit() else mds,
                "mcg": int(mcg) if str(mcg).isdigit() else mcg,
                "sta": int(sta) if str(sta).isdigit() else sta
            }
        }

        features.append(feature)
        kept += 1

# -----------------------------
# SAVE
# -----------------------------
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

with open(LATEST_FILE, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

# -----------------------------
# DONE
# -----------------------------
print(f"✅ Done -> {OUTPUT_FILE}")
print(f"✅ Updated latest -> {LATEST_FILE}")
print(f"⚡ Kept European strikes: {kept}")
print(f"❌ Skipped rows: {skipped}")
