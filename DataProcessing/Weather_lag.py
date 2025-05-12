import os
import pandas as pd

# Input files
base_dir = os.path.dirname(__file__)
all_years_path = os.path.join(base_dir, "results", "weather", "combined", "weather_2025.csv")
dec_path = os.path.join(base_dir, "results", "weather", "weather_2024_12.csv")
output_path = os.path.join(base_dir, "results", "weather", "combined", "weather_2025_with_lag.csv")

# Load data
print("📥 Loading all years for lag generation...")
df_main = pd.read_csv(all_years_path)
df_dec = pd.read_csv(dec_path)

# Combine Dec 2018 and main data temporarily for lag calculations
df = pd.concat([df_dec, df_main], ignore_index=True)

# Create datetime column
print("🧹 Preparing data...")
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
df = df.sort_values(by=["zone", "datetime"]).reset_index(drop=True)

# Variables to lag
weather_vars = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloudcover",
    "shortwave_radiation"
]

# Generate lag features
print("🛠️ Generating lag features...")
for lag in [1, 24]:
    for var in weather_vars:
        lag_col = f"{var}_lag{lag}"
        df[lag_col] = df.groupby("zone")[var].shift(lag)

# Drop Dec 2017 rows (anything before 2018)
df = df[df["date"] >= "2025-01-01"].copy()

# Drop temporary column and save
df = df.drop(columns=["datetime"])
df.to_csv(output_path, index=False)
print(f"✅ Lag-augmented dataset saved to: {output_path}")
