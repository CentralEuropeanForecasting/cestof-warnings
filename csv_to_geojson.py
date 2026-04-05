import csv
import json
from datetime import datetime

INPUT_FILE = "2026-04-05.csv"
OUTPUT_FILE = "2026-04-05_europe.geojson"

# Europe-ish bounds
MIN_LAT = 34.0
MAX_LAT = 72.0
MIN_LON = -25.0
MAX_LON = 45.0

def is_in_europe(lat, lon):
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON

features = []
kept = 0
skipped = 0

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 7:
            skipped += 1
            continue

        lat, lon, strike_time, server, mds, mcg, sta = row

        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            skipped += 1
            continue

        if not is_in_europe(lat, lon):
            skipped += 1
            continue

        hour = None
        try:
            hour = datetime.fromisoformat(strike_time).hour
        except ValueError:
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

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

print(f"Done -> {OUTPUT_FILE}")
print(f"Kept European strikes: {kept}")
print(f"Skipped non-European/invalid rows: {skipped}")