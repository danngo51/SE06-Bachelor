import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

CSV_PATH = "ml_models/informer/results/DK1-24-normalized.csv"
SEQ_LEN = 168
LABEL_LEN = 24
PRED_LEN = 24
BATCH_SIZE = 32

FEATURES = [
    "hour", "day", "weekday", "month", "weekend", "season",
    "ep_lag_1", "ep_lag_24", "ep_lag_168", "ep_rolling_mean_24",
    "dahtl_totalLoadValue", "dahtl_lag_1h", "dahtl_lag_24h", "dahtl_lag_168h",
    "dahtl_rolling_mean_24h", "dahtl_rolling_mean_168h",
    "atl_totalLoadValue", "atl_lag_1h", "atl_lag_24h", "atl_lag_168h",
    "atl_rolling_mean_24h", "atl_rolling_mean_168h",
    "temperature_2m", "wind_speed_10m", "wind_direction_10m", "cloudcover", "shortwave_radiation",
    "temperature_2m_lag1", "wind_speed_10m_lag1", "wind_direction_10m_lag1", "cloudcover_lag1", "shortwave_radiation_lag1",
    "temperature_2m_lag24", "wind_speed_10m_lag24", "wind_direction_10m_lag24", "cloudcover_lag24", "shortwave_radiation_lag24",
    "temperature_2m_lag168", "wind_speed_10m_lag168", "wind_direction_10m_lag168", "cloudcover_lag168", "shortwave_radiation_lag168",
    "ftc_DE_LU", "ftc_DK1", "ftc_GB", "ftc_NL", "Natural_Gas_price_EUR", "Natural_Gas_price_EUR_lag_1d", "Natural_Gas_price_EUR_lag_7d", 
    "Natural_Gas_rolling_mean_24h", "Coal_price_EUR", "Coal_price_EUR_lag_1d", "Coal_price_EUR_lag_7d", "Coal_rolling_mean_7d", 
    "Oil_price_EUR", "Oil_price_EUR_lag_1d", "Oil_price_EUR_lag_7d", "Oil_rolling_mean_7d", "Carbon_Emission_price_EUR", 
    "Carbon_Emission_price_EUR_lag_1d", "Carbon_Emission_price_EUR_lag_7d", "Carbon_Emission_rolling_mean_7d", "Price[Currency/MWh]"
]

TARGET = "Price[Currency/MWh]"

df = pd.read_csv(CSV_PATH, nrows=1)
csv_columns = set(df.columns) - {"date", "MapCode", "Price[Currency/MWh]"}
feature_set = set(FEATURES)

missing_from_features = csv_columns - feature_set

print("✅ Columns in CSV but not in FEATURES:", missing_from_features)
print("✅ Number of features in FEATURES:", len(FEATURES))


def load_training_batch():
    print("Number of features:", len(FEATURES))

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df.rename(columns={"date": "timestamp"}, inplace=True)
    df = df.dropna()

    full_data = df[FEATURES].values
    target_data = df[TARGET].values


    scaler = StandardScaler()
    full_data = scaler.fit_transform(full_data)
    print("Input shape:", full_data.shape) 

    time_features = pd.DataFrame({
    "month": df["timestamp"].dt.month,
    "day": df["timestamp"].dt.day,
    "weekday": df["timestamp"].dt.weekday,
    "hour": df["timestamp"].dt.hour
    }).values

    num_samples = len(df) - (SEQ_LEN + PRED_LEN)
    for i in range(0, num_samples - BATCH_SIZE, BATCH_SIZE):
        x_enc_batch, x_mark_enc_batch, x_dec_batch, x_mark_dec_batch, y_batch = [], [], [], [], []

        for j in range(BATCH_SIZE):
            start = i + j
            end = start + SEQ_LEN

            x_enc = full_data[start:end]
            x_mark_enc = time_features[start:end]

            x_dec = np.zeros((PRED_LEN, full_data.shape[1]))
            x_mark_dec = time_features[end:end + PRED_LEN]

            y = target_data[end:end + PRED_LEN]

            x_enc_batch.append(x_enc)
            x_mark_enc_batch.append(x_mark_enc)
            x_dec_batch.append(x_dec)
            x_mark_dec_batch.append(x_mark_dec)
            y_batch.append(y)

        yield (
            torch.tensor(x_enc_batch, dtype=torch.float32),
            torch.tensor(x_mark_enc_batch, dtype=torch.float32),
            torch.tensor(x_dec_batch, dtype=torch.float32),
            torch.tensor(x_mark_dec_batch, dtype=torch.float32),
            torch.tensor(y_batch, dtype=torch.float32)
        )


def load_input_sample(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.rename(columns={"date": "timestamp"}, inplace=True)
    df = df.dropna()

    x = df[FEATURES].values
    x = StandardScaler().fit_transform(x)

    time_features = pd.DataFrame({
    "month": df["timestamp"].dt.month,
    "day": df["timestamp"].dt.day,
    "weekday": df["timestamp"].dt.weekday,
    "hour": df["timestamp"].dt.hour
    }).values

    x_enc = torch.tensor(x[:SEQ_LEN], dtype=torch.float32).unsqueeze(0)
    x_mark_enc = torch.tensor(time_features[:SEQ_LEN], dtype=torch.float32).unsqueeze(0)

    x_dec = torch.zeros((1, PRED_LEN, len(FEATURES)))
    x_mark_dec = torch.tensor(time_features[SEQ_LEN:SEQ_LEN+PRED_LEN], dtype=torch.float32).unsqueeze(0)

    return x_enc, x_mark_enc, x_dec, x_mark_dec



