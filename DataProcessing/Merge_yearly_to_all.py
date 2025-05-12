import os
import pandas as pd

# Directory containing yearly CSV files
yearly_dir = os.path.join(os.path.dirname(__file__), "results", "weather", "combined")
output_path = os.path.join(yearly_dir, "weather_all_years.csv")

# Years to include
years = range(2018, 2025)
all_years = []

print("\n🔄 Combining all years into one file...")

for year in years:
    filename = f"weather_{year}.csv"
    filepath = os.path.join(yearly_dir, filename)

    if os.path.exists(filepath):
        print(f"  ➕ Adding {filename}")
        df = pd.read_csv(filepath)
        all_years.append(df)
    else:
        print(f"  ⚠️ Missing: {filename}")

if all_years:
    combined_df = pd.concat(all_years, ignore_index=True)

    # Sort by zone first, then date and hour
    combined_df = combined_df.sort_values(by=["zone", "date", "hour"])

    # Save combined file
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Combined file saved to: {output_path}")
else:
    print("❌ No yearly files found to combine.")
