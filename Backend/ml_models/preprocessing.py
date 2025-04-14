import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    features = ["price", "load", "wind", "solar", "temp"]
    x_enc = df[features].values
    x_enc = StandardScaler().fit_transform(x_enc)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    x_mark_enc = df[["hour", "dayofweek", "month"]].values

    x_dec = torch.zeros((1, 24, len(features)))
    x_mark_dec = torch.zeros((1, 24, 3))

    x_enc = torch.tensor(x_enc, dtype=torch.float32).unsqueeze(0)
    x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32).unsqueeze(0)

    return x_enc, x_mark_enc, x_dec, x_mark_dec



def load_training_batch(csv_path="data/train_2019_2023.csv", seq_len=168, pred_len=24, batch_stride=1):
    """
    Generator that yields training samples from 2019–2023 data.
    Each sample = 168h encoder input → predict average price over next 24h
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    features = ["price", "load", "wind", "solar", "temp"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Create time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    x_all = df[features].values
    x_time = df[["hour", "dayofweek", "month"]].values
    price_series = df["price"].values

    total_length = len(df)
    max_start = total_length - (seq_len + pred_len)

    for start in range(0, max_start, batch_stride):
        end_enc = start + seq_len
        end_dec = end_enc + pred_len

        # Skip if we cross into 2024
        if pd.to_datetime(df["timestamp"].iloc[end_dec - 1]).year > 2023:
            break

        x_enc = torch.tensor(x_all[start:end_enc], dtype=torch.float32).unsqueeze(0)
        x_mark_enc = torch.tensor(x_time[start:end_enc], dtype=torch.float32).unsqueeze(0)

        # Informer-style dummy decoder input (we won't use it in GRU)
        x_dec = torch.zeros((1, pred_len, len(features)))
        x_mark_dec = torch.zeros((1, pred_len, 3))

        # Target: average price over next 24h
        y = torch.tensor([[price_series[end_enc:end_dec].mean()]], dtype=torch.float32)

        yield x_enc, x_mark_enc, x_dec, x_mark_dec, y
