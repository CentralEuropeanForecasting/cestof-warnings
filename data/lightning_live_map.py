#!/usr/bin/env python

import os
import csv
import time
from datetime import date, timedelta
import requests
#from geopy.geocoders import Nominatim

UPDATE_EVERY_SECONDS = 30

MIN_LAT = 34.0
MAX_LAT = 72.0
MIN_LON = -25.0
MAX_LON = 45.0

def is_in_europe(latitude, longitude):
    return MIN_LAT <= float(latitude) <= MAX_LAT and MIN_LON <= float(longitude) <= MAX_LON

### Intended to be run once per minute. 
### Request recent lightning strikes and save them into a csv file
def main():
    os.makedirs("data", exist_ok=True)

    while True:
        # Read the existing CSV file from today and yesterday (for edge cases) and remember the 
        # time of each strike, which is used as a unique identifier.
        recorded_times = set() 
        today = date.today()
        yesterday = today - timedelta(days = 1)
        today_filepath = os.path.join("data", "%s.csv" % today.strftime('%Y-%m-%d'))
        yesterday_filepath = os.path.join("data", "%s.csv" % yesterday.strftime('%Y-%m-%d'))

        for path in [today_filepath, yesterday_filepath]:
            if os.path.exists(path):
                with open(path, newline='') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        if len(row) < 7:
                            continue
                        [latitude, longitude, strike_time, server, mds, mcg, sta] = row
                        if strike_time not in recorded_times:
                            recorded_times.add(strike_time)
        
        # Get recent lightning strikes from Blitzortung server
        response = requests.get(
            "https://map.blitzortung.org/GEOjson/getjson.php?f=s&n=00",
            headers={"Referer": "https://map.blitzortung.org/"},
            timeout=15
        )
        strikes = response.json()

        added = 0

        # Append them to a local daily csv file if they're not already recorded
        with open(today_filepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            for strike in strikes:
                # MDS (maximal deviation span in nanoseconds)
                # MCG (maximal circular gap in degree)
                # sta = ??? Maybe number of stations that saw it? Maybe strength? I'm not sure yet..
                (longitude, latitude, strike_time, server, mds, mcg, sta) = strike

                try:
                    if not is_in_europe(latitude, longitude):
                        continue
                except:
                    continue
                
                if strike_time not in recorded_times:
                    csvwriter.writerow([latitude, longitude, strike_time, server, mds, mcg, sta])
                    recorded_times.add(strike_time)
                    added += 1

        print("Added %s new European strikes to %s" % (added, today_filepath))
        time.sleep(UPDATE_EVERY_SECONDS)
        

# geolocator = Nominatim(user_agent="geoapiExercises")
#
# countries = []
#
# for strike in strikes:
#     # MDS (maximal deviation span in nanoseconds)
#     # MCG (maximal circular gap in degree)
#     # sta = ??? Maybe number of stations that saw it? Maybe strength? I'm not sure yet..
#     (longitude, latitude, date, server, mds, mcg, sta) = strike
#     location = geolocator.reverse("%s,%s" % (latitude, longitude))
#     if (location is not None):
#         address = location.raw['address']
#         country = address.get('country')
#         if (country is not None):
#             if (country not in countries):
#                 countries.append(country)
#
# print(countries)

if __name__ == "__main__":
    main()