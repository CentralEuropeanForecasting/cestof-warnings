import csv
import json
import os
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
one_hour_ago = now - timedelta(hours=1)

INPUT_FILES = [
    os.path.join("data", (now - timedelta(days=1)).strftime("%Y-%m-%d") + ".csv"),
    os.path.join("data", now.strftime("%Y-%m-%d") + ".csv"),
]

OUTPUT_FILE = os.path.join("data", "last_1_hour_europe.geojson")
LATEST_FILE = os.path.join("data", "latest_europe.geojson")

MIN_LAT = 34.0
MAX_LAT = 72.0
MIN_LON = -25.0
MAX_LON = 45.0

def is_in_europe(lat, lon):
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON

def parse_strike_time(value):
    s = str(value).strip()
    s = s.replace("Z", "+00:00")

    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        if "." in s:
            base, frac = s.split(".", 1)
            frac_digits = ""
            tz_part = ""
            for ch in frac:
                if ch.isdigit():
                    frac_digits += ch
                else:
                    tz_part = frac[len(frac_digits):]
                    break
            frac_digits = (frac_digits[:6]).ljust(6, "0")
            s2 = f"{base}.{frac_digits}{tz_part}"
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    return None

features = []
seen = set()
kept = 0
skipped = 0

for input_file in INPUT_FILES:
    if not os.path.exists(input_file):
        continue

    with open(input_file, newline="", encoding="utf-8") as f:
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

            dt = parse_strike_time(strike_time)
            if dt is None:
                skipped += 1
                continue

            if not (one_hour_ago <= dt <= now):
                skipped += 1
                continue

            strike_id = f"{lat},{lon},{strike_time}"
            if strike_id in seen:
                continue
            seen.add(strike_id)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "strike_time": dt.isoformat(),
                    "hour": dt.hour,
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

with open(LATEST_FILE, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

print(f"Now (UTC): {now.isoformat()}")
print(f"1 hour ago (UTC): {one_hour_ago.isoformat()}")
print(f"Done -> {OUTPUT_FILE}")
print(f"Done -> {LATEST_FILE}")
print(f"Kept European strikes in last 1 hour: {kept}")
print(f"Skipped invalid/out-of-window rows: {skipped}")
