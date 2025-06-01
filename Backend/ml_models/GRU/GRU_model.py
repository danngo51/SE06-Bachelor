import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pathlib
import joblib
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        regressor_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.regressor = nn.Linear(regressor_input_dim, output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        if self.gru.bidirectional:
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            hidden_concat = torch.cat((last_forward, last_backward), dim=1)
            return self.regressor(hidden_concat)
        return self.regressor(hidden[-1])

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class GRUModelTrainer:
    def __init__(self, mapcode="DK1", seq_len=168, pred_len=24):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 128
        self.num_layers = 2
        self.bidirectional = True
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 20
        self.patience = 3
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"
        self.gru_dir.mkdir(exist_ok=True)

    def create_features(self, df):
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        for lag in [1, 24, 168]:
            df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)
        for window in [24, 168]:
            df[f'price_roll_mean_{window}'] = df['Electricity_price_MWh'].rolling(window).mean()
        return df.dropna()

    def prepare_sequences(self, df):
        feature_cols = [col for col in df.columns if col not in ['date', 'Electricity_price_MWh']]
        features = df[feature_cols].values
        target = df['Electricity_price_MWh'].values
        sequences, targets = [], []
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            sequences.append(features[i:i + self.seq_len])
            targets.append(target[i + self.seq_len:i + self.seq_len + self.pred_len])
        return np.array(sequences), np.array(targets), feature_cols

    def train(self, train_file, val_file=None):
        train_df = pd.read_csv(train_file, parse_dates=['date'])
        train_df = self.create_features(train_df)
        train_sequences, train_targets, feature_cols = self.prepare_sequences(train_df)
        self.input_dim = train_sequences.shape[-1]
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        train_sequences = self.scaler_features.fit_transform(train_sequences.reshape(-1, self.input_dim)).reshape(train_sequences.shape)
        train_targets = self.scaler_target.fit_transform(train_targets)
        train_dataset = TimeSeriesDataset(train_sequences, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model = GRUModel(self.input_dim, self.hidden_dim, self.num_layers, self.pred_len, self.bidirectional).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences, batch_targets = batch_sequences.to(self.device), batch_targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(train_loader))
        return train_losses

    def predict(self, data):
        data = self.create_features(data)
        sequences, _, _ = self.prepare_sequences(data)
        sequences = self.scaler_features.transform(sequences.reshape(-1, self.input_dim)).reshape(sequences.shape)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequences_tensor).cpu().numpy()
        return self.scaler_target.inverse_transform(predictions)

    def save_model(self):
        torch.save(self.model.state_dict(), str(self.gru_dir / "gru_model.pth"))
        joblib.dump(self.scaler_features, str(self.gru_dir / "scaler_features.pkl"))
        joblib.dump(self.scaler_target, str(self.gru_dir / "scaler_target.pkl"))

    def load_model(self):
        self.model = GRUModel(self.input_dim, self.hidden_dim, self.num_layers, self.pred_len, self.bidirectional).to(self.device)
        self.model.load_state_dict(torch.load(str(self.gru_dir / "gru_model.pth"), map_location=self.device))
        self.scaler_features = joblib.load(str(self.gru_dir / "scaler_features.pkl"))
        self.scaler_target = joblib.load(str(self.gru_dir / "scaler_target.pkl"))