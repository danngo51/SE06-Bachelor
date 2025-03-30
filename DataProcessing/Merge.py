import os
import pandas as pd

# Directory containing monthly CSV files
monthly_dir = os.path.join(os.path.dirname(__file__), "results", "weather")
output_dir = os.path.join(monthly_dir, "combined")
os.makedirs(output_dir, exist_ok=True)

# Years to combine
years = range(2019, 2024)

for year in years:
    print(f"Combining files for year {year}...")
    yearly_dfs = []

    for month in range(1, 13):
        filename = f"weather_{year}_{month:02d}.csv"
        filepath = os.path.join(monthly_dir, filename)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            yearly_dfs.append(df)
        else:
            print(f"⚠️ File not found: {filename}")

    if yearly_dfs:
        combined_df = pd.concat(yearly_dfs)
        output_path = os.path.join(output_dir, f"weather_{year}.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Year {year} saved to {output_path}")
    else:
        print(f"❌ No data found for year {year}")