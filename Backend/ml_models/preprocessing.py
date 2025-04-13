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
