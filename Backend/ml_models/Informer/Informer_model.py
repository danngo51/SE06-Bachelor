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


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, seq_len=168, pred_len=24):
        super().__init__()
        self.value_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_len + pred_len, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.value_emb(x) + self.pos_emb(pos))

class Informer(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, e_layers=2, d_layers=1, dropout=0.2, seq_len=168, label_len=48, pred_len=24):
        super().__init__()
        self.enc_emb = DataEmbedding(input_dim, d_model, seq_len, pred_len)
        self.dec_emb = DataEmbedding(1, d_model, seq_len, pred_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=e_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=d_layers
        )
        self.proj = nn.Linear(d_model, 1)

    def forward(self, enc_x, dec_y=None):
        if dec_y is None:
            dec_y = torch.zeros((enc_x.shape[0], self.label_len + self.pred_len), device=enc_x.device)
            if self.label_len > 0 and enc_x.shape[1] >= self.label_len:
                last_values = enc_x[:, -self.label_len:, -1]
                dec_y[:, :self.label_len] = last_values
        enc_out = self.encoder(self.enc_emb(enc_x))
        dec_out = self.decoder(self.dec_emb(dec_y.unsqueeze(-1)), enc_out)
        return self.proj(dec_out).squeeze(-1)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len, label_len, pred_len, scaler=None):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        data = df[feature_cols + [target_col]].values.astype(np.float32)
        self.scaler = scaler or StandardScaler().fit(data)
        data = self.scaler.transform(data)
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.samples = [
            (self.features[i:i+seq_len], self.targets[i+seq_len-label_len:i+seq_len+pred_len])
            for i in range(len(data) - seq_len - pred_len + 1)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc_x, dec_y = self.samples[idx]
        return {'enc_x': torch.from_numpy(enc_x), 'dec_y': torch.from_numpy(dec_y)}

class InformerModelTrainer:
    def __init__(self, mapcode="DK1", seq_len=168, label_len=48, 
                pred_len=24, batch_size=32, learning_rate=1e-4,
                epochs=50, early_stop_patience=8, device=None):
        
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.informer_dir = self.data_dir / "informer"
        self.feature_file = self.data_dir / "feature" / "features.csv"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.informer_dir, exist_ok=True)

    def train(self, data_path=None):
        data_path = data_path or self.data_dir / f"{self.mapcode}_full_data_2018_2024.csv"
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date').dropna()

        # Load top features from features.csv
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")
        feature_cols = pd.read_csv(self.feature_file)['Feature'].tolist()

        # Ensure the dataset includes 'date', top features, and target column
        required_cols = ['date'] + feature_cols + ['Electricity_price_MWh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        target_col = 'Electricity_price_MWh'
        train_df = df.loc['2018-01-01':'2022-12-31']
        val_df = df.loc['2023-01-01':'2023-12-31']
        test_df = df.loc['2024-01-01':'2024-12-31']
        train_set = TimeSeriesDataset(train_df, feature_cols, target_col, self.seq_len, self.label_len, self.pred_len)
        val_set = TimeSeriesDataset(val_df, feature_cols, target_col, self.seq_len, self.label_len, self.pred_len, scaler=train_set.scaler)
        test_set = TimeSeriesDataset(test_df, feature_cols, target_col, self.seq_len, self.label_len, self.pred_len, scaler=train_set.scaler)
        joblib.dump(train_set.scaler, self.informer_dir / "scaler.pkl")
        joblib.dump(feature_cols, self.informer_dir / "feature_columns.pkl")
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        model = Informer(input_dim=len(feature_cols), seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        best_val, wait = float('inf'), 0
        train_losses, val_losses = [], []

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = sum(
                criterion(model(batch['enc_x'].to(self.device), batch['dec_y'].to(self.device)[:, :self.label_len+self.pred_len])[:, -self.pred_len:], batch['dec_y'].to(self.device)[:, -self.pred_len:]).item()
                for batch in train_loader
            ) / len(train_loader)
            train_losses.append(train_loss)
            model.eval()
            val_loss = sum(
                criterion(model(batch['enc_x'].to(self.device), batch['dec_y'].to(self.device)[:, :self.label_len+self.pred_len])[:, -self.pred_len:], batch['dec_y'].to(self.device)[:, -self.pred_len:]).item()
                for batch in val_loader
            ) / len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val, wait = val_loss, 0
                torch.save(model.state_dict(), self.informer_dir / 'best_informer.pt')
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(self.informer_dir / 'best_informer.pt', map_location=self.device))
        model.eval()
        all_preds, all_targs = [], []
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                enc_x = batch['enc_x'].to(self.device)
                dec_y = batch['dec_y'].to(self.device)
                out = model(enc_x, dec_y[:, :self.label_len+self.pred_len])
                test_loss += criterion(out[:, -self.pred_len:], dec_y[:, -self.pred_len:]).item()
                all_preds.append(out[:, -self.pred_len:].cpu().numpy())
                all_targs.append(dec_y[:, -self.pred_len:].cpu().numpy())
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")

        all_preds = np.vstack(all_preds)
        all_targs = np.vstack(all_targs)
        mean, var = train_set.scaler.mean_[-1], train_set.scaler.var_[-1]
        std = np.sqrt(var)
        real_preds = all_preds * std + mean
        real_targs = all_targs * std + mean
        mse = mean_squared_error(real_targs.flatten(), real_preds.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_targs.flatten(), real_preds.flatten())
        r2 = r2_score(real_targs.flatten(), real_preds.flatten())
        metrics_data = {'Metric': ['RMSE', 'MAE', 'R²'], 'Value': [rmse, mae, r2]}
        pd.DataFrame(metrics_data).to_csv(self.informer_dir / "metrics.csv", index=False)

        return {
            "model_path": str(self.informer_dir / 'best_informer.pt'),
            "scaler_path": str(self.informer_dir / 'scaler.pkl'),
            "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        }