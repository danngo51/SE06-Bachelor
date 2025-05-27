# forecast_day.py
# Standalone script to load pre-trained Informer model and forecast 24h ahead prices for a given date,
# include actual prices and percentage error in CSV output.

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Informer_model import Informer

# -----------------------------
# Configuration & CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Forecast day-ahead electricity prices with error metrics")
    p.add_argument("--date", type=str, required=True,
                   help="Date to forecast (YYYY-MM-DD), e.g. 2025-03-05")
    p.add_argument("--mapcode", type=str, default="DK1",
                   help="Market mapcode, default DK1")
    return p.parse_args()

args = parse_args()
mapcode = args.mapcode
forecast_date = pd.to_datetime(args.date)

# Paths
template_path = f'/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/CombinedDatasetPerCountry/{mapcode}'
output_path = f'/Volumes/SSD/SEBachelor/entso-e data/InformerModel/Results/{mapcode}'
train_path = os.path.join(template_path, f"{mapcode}_full_data_2018_2024.csv")
predict_path = os.path.join(template_path, f"2025/{mapcode}_full_data_2025.csv")
model_path = os.path.join(output_path, 'best_informer.pt')

# Model / scaler parameters
SEQ_LEN = 168
LABEL_LEN = 24
PRED_LEN = 24
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 1) Load and fit scaler on historical data
print(f"Loading training data for scaler: {train_path}")
df_train = pd.read_csv(train_path, parse_dates=['date'], index_col='date')
feature_cols = [c for c in df_train.columns if c != 'Electricity_price_MWh']
data_train = df_train[feature_cols + ['Electricity_price_MWh']].values.astype(np.float32)
scaler = StandardScaler().fit(data_train)

# 2) Load 2025 data and check date
print(f"Loading 2025 data: {predict_path}")
df_pred = pd.read_csv(predict_path, parse_dates=['date'], index_col='date')
if forecast_date not in df_pred.index:
    print(f"Error: {forecast_date.date()} not in data index")
    sys.exit(1)

# 3) Scale features & extract true prices
data_pred = df_pred[feature_cols + ['Electricity_price_MWh']].values.astype(np.float32)
data_scaled = scaler.transform(data_pred)
features_scaled = data_scaled[:, :-1]
true_prices = df_pred['Electricity_price_MWh'].values  # real prices

i = df_pred.index.get_loc(forecast_date)
if i < SEQ_LEN:
    print(f"Insufficient history before {forecast_date.date()}")
    sys.exit(1)

# 4) Build encoder & decoder inputs
enc_seq = features_scaled[i-SEQ_LEN:i]
enc_x = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
hist_scaled = data_scaled[i-LABEL_LEN:i, -1]
# decoder input: last labels + zeros placeholder
dec_in = np.concatenate([hist_scaled, np.zeros(PRED_LEN, dtype=np.float32)])
dec_y = torch.tensor(dec_in, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# 5) Load model and forecast
print(f"Loading model: {model_path}")
model = Informer(input_dim=len(feature_cols))
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE).eval()
print(f"Forecasting for {forecast_date.date()}")
with torch.no_grad():
    out = model(enc_x, dec_y)
preds_scaled = out.cpu().numpy().flatten()[-PRED_LEN:]
# inverse-scale
mean, var = scaler.mean_[-1], scaler.var_[-1]
std = np.sqrt(var)
preds_real = preds_scaled * std + mean

# 6) Prepare results DataFrame
dates = pd.date_range(start=forecast_date, periods=PRED_LEN, freq='H')
actuals = true_prices[i:i+PRED_LEN]
# percentage error: abs(pred-actual)/actual*100, safe divide
pct_error = np.where(actuals!=0, np.abs(preds_real-actuals)/np.abs(actuals)*100, np.nan)

df_out = pd.DataFrame({
    'forecast_datetime': dates,
    'forecast_price_MWh': preds_real,
    'actual_price_MWh': actuals,
    'pct_error': pct_error
})

out_csv = os.path.join(output_path, f"{mapcode}_forecast_{forecast_date.date()}.csv")
df_out.to_csv(out_csv, index=False)
print(f"Saved forecast CSV with actuals & pct_error to {out_csv}")

# 7) Plot forecast vs actual
plt.figure(figsize=(10,4))
plt.plot(dates, actuals, label='Actual')
plt.plot(dates, preds_real, label='Forecast', linestyle='--')
plt.title(f"Forecast vs Actual on {forecast_date.date()}")
plt.xlabel('DateTime')
plt.ylabel('Price (EUR/MWh)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
out_png = os.path.join(output_path, f"{mapcode}_forecast_plot_{forecast_date.date()}.png")
plt.savefig(out_png)
print(f"Saved plot to {out_png}")
plt.show()