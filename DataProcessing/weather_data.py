import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


import os

# Create results/weather directory relative to this script
output_dir = os.path.join(os.path.dirname(__file__), "results", "weather")
os.makedirs(output_dir, exist_ok=True)

# Setup cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=0)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define DK1 and DK2 with coordinates
zones = {
    "DK1": {
        "lat": 56.1629,
        "lon": 10.2039,
        "timezone": "Europe/Copenhagen"
    },
    "DK2": {
        "lat": 55.6761,
        "lon": 12.5683,
        "timezone": "Europe/Copenhagen"
    }
}

season_map = {
    12: 0, 1: 0, 2: 0,   # Winter
    3: 1, 4: 1, 5: 1,    # Spring
    6: 2, 7: 2, 8: 2,    # Summer
    9: 3, 10: 3, 11: 3   # Fall
}


# Date range for test (January 2019)
start_date = "2019-01-01"
end_date = "2019-01-03"

# Weather variables to fetch
weather_vars = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloudcover",  # note: in API it's cloudcover, not cloud_cover
    "shortwave_radiation"
]

all_data = []

for zone, info in zones.items():
    lat = info["lat"]
    lon = info["lon"]
    timezone = info["timezone"]
    
    print(f"Fetching data for {zone}...")

    params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": weather_vars,
    "timezone": timezone
    }
    
    responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response = responses[0]
    hourly = response.Hourly()

    # Extract values in the same order as requested
    time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    df = pd.DataFrame({
        "zone": zone,
        "timestamp": time,
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(2).ValuesAsNumpy(),
        "cloudcover": hourly.Variables(3).ValuesAsNumpy(),
        "shortwave_radiation": hourly.Variables(4).ValuesAsNumpy(),
    
    })

    # Convert to local time
    df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)

    # Add time columns and drop original timestamp
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday  # 0 = Monday
    df["month"] = df["timestamp"].dt.month
    df["weekend"] = df["weekday"].isin([4, 5, 6]).astype(int)  # Friday, Saturday, Sunday
    df["season"] = df["month"].map(season_map)

    df = df.drop(columns=["timestamp"])  # Drop original timestamp

    # Reorder columns (time features first)
    cols = ["zone", "date", "hour", "day", "weekday", "month", "weekend", "season",
            "temperature_2m", "wind_speed_10m", "wind_direction_10m",
            "cloudcover", "shortwave_radiation"]
    
    df = df[cols]

    all_data.append(df)


# Merge DK1 and DK2
combined_df = pd.concat(all_data)

output_path = os.path.join(output_dir, "weather_2019_01.csv")
combined_df.to_csv(output_path, index=False)


print("✅ Saved DK1 + DK2 weather data to weather_2019_01.csv")



