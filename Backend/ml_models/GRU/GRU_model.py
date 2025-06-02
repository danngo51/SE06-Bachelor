import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from ml_models.GRU.GRU_model import GRUModel

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # take the last time step
        return out



class GRUTrainer:
    def __init__(self, mapcode="DK1", seq_len=168, pred_len=24, lr=0.001, device=None):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.model = None
        self.num_epochs = 50
        self.patience = 10

        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.gru_dir, exist_ok=True)

    def load_dataset(self, csv_path: str):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.drop(columns=['date'])  # keep only numeric features
        scaler = StandardScaler()
        values = scaler.fit_transform(df.values)

        X, y = [], []
        for i in range(len(values) - self.seq_len - self.pred_len + 1):
            X.append(values[i:i + self.seq_len])
            y.append(values[i + self.seq_len:i + self.seq_len + self.pred_len, 0])  # predicting the first feature

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    def train(self, train_file: str):
        train_loader, scaler = self.load_dataset(train_file)
        input_size = next(iter(train_loader))[0].shape[2]
        output_size = self.pred_len

        self.model = GRUModel(input_size, hidden_size=64, num_layers=2, output_size=output_size).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping")
                    break

        # Final evaluation on training set
        print("Evaluating on training set...")

        self.model.eval()
        all_preds, all_targs = [], []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                output = self.model(x_batch)
                all_preds.append(output.cpu().numpy())
                all_targs.append(y_batch.cpu().numpy())

        preds = np.vstack(all_preds)
        targs = np.vstack(all_targs)

        mean = scaler.mean_[0]
        std = np.sqrt(scaler.var_[0])
        preds_real = preds * std + mean
        targs_real = targs * std + mean

        mse = mean_squared_error(targs_real, preds_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targs_real, preds_real)
        r2 = r2_score(targs_real, preds_real)

        metrics_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "R²"],
            "Value": [rmse, mae, r2]
        })

        metrics_path = self.gru_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
        

    def save_model(self):
        model_path = self.gru_dir / "gru_model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
