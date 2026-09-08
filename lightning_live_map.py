#!/usr/bin/env python3

import os
import csv
from datetime import datetime, timezone, timedelta
import requests

DATA_DIR = "data"
BLITZ_URL = "https://map.blitzortung.org/GEOjson/getjson.php?f=s&n=00"
HEADERS = {"Referer": "https://map.blitzortung.org/"}


def parse_strike_time(value):
    """Parse Blitzortung timestamp and treat it as UTC."""
    value = str(value).strip()

    # Python only supports up to 6 fractional-second digits.
    if "." in value:
        main, frac = value.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())[:6].ljust(6, "0")
        value = f"{main}.{frac}"

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    raise ValueError(f"Unrecognised strike timestamp: {value}")


def load_recorded_keys(paths):
    """Load existing strike identities from all relevant daily CSV files."""
    recorded = set()

    for path in paths:
        if not os.path.exists(path):
            continue

        with open(path, newline="", encoding="utf-8") as csvfile:
            for row in csv.reader(csvfile):
                if len(row) < 7:
                    continue
                latitude, longitude, strike_time, server, mds, mcg, sta = row[:7]
                recorded.add((latitude, longitude, strike_time, server))

    return recorded


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Use UTC consistently.
    now_utc = datetime.now(timezone.utc)
    dates_to_load = {
        now_utc.date() - timedelta(days=1),
        now_utc.date(),
        now_utc.date() + timedelta(days=1),
    }

    paths_to_load = [
        os.path.join(DATA_DIR, f"{d:%Y-%m-%d}.csv")
        for d in dates_to_load
    ]
    recorded = load_recorded_keys(paths_to_load)

    response = requests.get(
        BLITZ_URL,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    strikes = response.json()

    # Group incoming strikes by THEIR OWN UTC DATE, not the runner's date.
    rows_by_path = {}

    for strike in strikes:
        if len(strike) != 7:
            continue

        longitude, latitude, strike_time, server, mds, mcg, sta = strike

        try:
            strike_dt = parse_strike_time(strike_time)
        except ValueError:
            continue

        strike_date = strike_dt.date()
        filepath = os.path.join(DATA_DIR, f"{strike_date:%Y-%m-%d}.csv")

        row = [
            str(latitude),
            str(longitude),
            str(strike_time),
            str(server),
            str(mds),
            str(mcg),
            str(sta),
        ]

        # More robust than timestamp alone: same timestamp from different
        # coordinates/servers will not accidentally suppress a real strike.
        key = (row[0], row[1], row[2], row[3])

        if key in recorded:
            continue

        recorded.add(key)
        rows_by_path.setdefault(filepath, []).append(row)

    total_added = 0

    for filepath, rows in rows_by_path.items():
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerows(rows)
        total_added += len(rows)
        print(f"Added {len(rows)} strikes to {filepath}")

    print(f"Finished. Added {total_added} new strikes.")


if __name__ == "__main__":
    main()
