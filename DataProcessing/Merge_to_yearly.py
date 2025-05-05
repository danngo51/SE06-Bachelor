import os
import pandas as pd

# Directory containing monthly CSV files
monthly_dir = os.path.join(os.path.dirname(__file__), "results", "weather")
output_dir = os.path.join(monthly_dir, "combined")
os.makedirs(output_dir, exist_ok=True)

# Years to combine
years = range(2024, 2025)

for year in years:
    print(f"\n🔄 Combining and cleaning year {year}...")
    yearly_dfs = []

    for month in range(1, 13):
        filename = f"weather_{year}_{month:02d}.csv"
        filepath = os.path.join(monthly_dir, filename)

        if os.path.exists(filepath):
            print(f"  ➕ Adding {filename}")
            df = pd.read_csv(filepath)
            df = df[df["date"].str.startswith(str(year))]  # Ensure only rows from this year are included
            yearly_dfs.append(df)
        else:
            print(f"  ⚠️ File not found: {filename}")

    if yearly_dfs:
        combined_df = pd.concat(yearly_dfs, ignore_index=True)

        # Remove duplicates caused by month overlap
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["zone", "date", "hour"])
        after = len(combined_df)
        print(f"  🧹 Removed {before - after} duplicate rows")

        # Sort by zone, then date, then hour
        combined_df = combined_df.sort_values(by=["zone", "date", "hour"])

        # Save cleaned, combined file
        output_path = os.path.join(output_dir, f"weather_{year}.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned year file: {output_path}")
    else:
        print(f"❌ No data found to combine for year {year}")
