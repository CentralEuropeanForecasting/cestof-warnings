#!/usr/bin/env python

import os
import csv
import time
from datetime import date, timedelta
import requests

UPDATE_EVERY_SECONDS = 30

MIN_LAT = 34.0
MAX_LAT = 72.0
MIN_LON = -25.0
MAX_LON = 45.0

def is_in_europe(latitude, longitude):
    return MIN_LAT <= float(latitude) <= MAX_LAT and MIN_LON <= float(longitude) <= MAX_LON

def update_once():
    os.makedirs("data", exist_ok=True)

    recorded_times = set()
    today = date.today()
    yesterday = today - timedelta(days=1)

    today_filepath = os.path.join("data", f"{today.strftime('%Y-%m-%d')}.csv")
    yesterday_filepath = os.path.join("data", f"{yesterday.strftime('%Y-%m-%d')}.csv")

    for path in [today_filepath, yesterday_filepath]:
        if os.path.exists(path):
            with open(path, newline="") as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if len(row) < 7:
                        continue
                    latitude, longitude, strike_time, server, mds, mcg, sta = row
                    recorded_times.add(strike_time)

    response = requests.get(
        "https://map.blitzortung.org/GEOjson/getjson.php?f=s&n=00",
        headers={"Referer": "https://map.blitzortung.org/"},
        timeout=15
    )
    response.raise_for_status()
    strikes = response.json()

    added = 0

    with open(today_filepath, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        for strike in strikes:
            longitude, latitude, strike_time, server, mds, mcg, sta = strike

            try:
                if not is_in_europe(latitude, longitude):
                    continue
            except Exception:
                continue

            if strike_time not in recorded_times:
                csvwriter.writerow([latitude, longitude, strike_time, server, mds, mcg, sta])
                recorded_times.add(strike_time)
                added += 1

    print(f"Added {added} new European strikes to {today_filepath}")

def main():
    while True:
        update_once()
        time.sleep(UPDATE_EVERY_SECONDS)

if __name__ == "__main__":
    if os.environ.get("RUN_ONCE") == "1":
        update_once()
    else:
        main()
