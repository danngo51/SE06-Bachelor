import pandas as pd
from pathlib import Path

# Use relative paths
base_path = Path(__file__).parent
data_path = base_path / "DK1-24-normalized.csv"
results_path = base_path / "results"
results_path.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(data_path, parse_dates=["date"])

# Split by years
train_df = df[df["date"].dt.year <= 2021]
val_df   = df[df["date"].dt.year == 2022]
test_df  = df[df["date"].dt.year == 2023]

# Save to results directory
train_df.to_csv(results_path / "train.csv", index=False)
val_df.to_csv(results_path / "val.csv", index=False)
test_df.to_csv(results_path / "test.csv", index=False)

print("✅ Split complete. Files saved in:", results_path)