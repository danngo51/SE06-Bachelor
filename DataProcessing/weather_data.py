import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from Weather.zones import zones
from datetime import datetime, timedelta
import os

# Create results/weather directory relative to this script
output_dir = os.path.join(os.path.dirname(__file__), "results", "weather")
os.makedirs(output_dir, exist_ok=True)

# Setup cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=0)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

season_map = {
    12: 0, 1: 0, 2: 0,   # Winter
    3: 1, 4: 1, 5: 1,    # Spring
    6: 2, 7: 2, 8: 2,    # Summer
    9: 3, 10: 3, 11: 3   # Fall
}

# Weather variables to fetch
weather_vars = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloudcover",
    "shortwave_radiation"
]

# Loop through all months from Jan 2019 to Dec 2023
for year in range(2019, 2024):
    for month in range(1, 13):
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"

        print(f"\n📆 Processing: {start_date} to {end_date}")

        all_data = []

        for zone, info in zones.items():
            timezone = info["timezone"]
            zone_dfs = []

            print(f"  🌍 Fetching data for {zone}...")

            for loc in info["locations"]:
                params = {
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": weather_vars,
                    "timezone": timezone
                }

                responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
                response = responses[0]
                hourly = response.Hourly()

                time = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )

                df = pd.DataFrame({
                    "timestamp": time,
                    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                    "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
                    "wind_direction_10m": hourly.Variables(2).ValuesAsNumpy(),
                    "cloudcover": hourly.Variables(3).ValuesAsNumpy(),
                    "shortwave_radiation": hourly.Variables(4).ValuesAsNumpy()
                })

                zone_dfs.append(df)

            # Average across locations
            merged = pd.concat(zone_dfs).groupby("timestamp").mean().reset_index()
            merged["timestamp"] = merged["timestamp"].dt.tz_convert(timezone)

            merged["zone"] = zone
            merged["date"] = merged["timestamp"].dt.strftime("%Y-%m-%d")
            merged["hour"] = merged["timestamp"].dt.hour
            merged["day"] = merged["timestamp"].dt.day
            merged["weekday"] = merged["timestamp"].dt.weekday
            merged["month"] = merged["timestamp"].dt.month
            merged["weekend"] = merged["weekday"].isin([4, 5, 6]).astype(int)
            merged["season"] = merged["month"].map(season_map)

            merged = merged.drop(columns=["timestamp"])

            cols = [
                "zone", "date", "hour", "day", "weekday", "month", "weekend", "season",
                "temperature_2m", "wind_speed_10m", "wind_direction_10m",
                "cloudcover", "shortwave_radiation"
            ]
            merged = merged[cols]
            all_data.append(merged)

        # Combine and export for the month
        combined_df = pd.concat(all_data)
        output_path = os.path.join(output_dir, f"weather_{year}_{month:02d}.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Saved {output_path}")
