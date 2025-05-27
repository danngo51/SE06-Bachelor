import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

# Ensure spawn method for macOS
multiprocessing.set_start_method('spawn', force=True)

# -----------------------------
# Configuration
# -----------------------------
mapcode = "DK1"
file_path = f'/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/CombinedDatasetPerCountry/{mapcode}'
output_path = f'/Volumes/SSD/SEBachelor/entso-e data/InformerModel/Results/{mapcode}'
DATA_PATH = os.path.join(file_path, f"{mapcode}_full_data_2018_2024.csv")
SEQ_LEN = 168
LABEL_LEN = 48
PRED_LEN = 24
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.set_num_threads(4)
os.makedirs(output_path, exist_ok=True)

# -----------------------------
# Dataset Definition
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len, label_len, pred_len):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        data = df[feature_cols + [target_col]].values.astype(np.float32)
        self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.samples = []
        for i in range(len(data) - seq_len - pred_len + 1):
            enc_x = self.features[i:i+seq_len]
            dec_y = self.targets[i+seq_len-label_len:i+seq_len+pred_len]
            self.samples.append((enc_x, dec_y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        enc_x, dec_y = self.samples[idx]
        return {'enc_x': torch.from_numpy(enc_x), 'dec_y': torch.from_numpy(dec_y)}

# -----------------------------
# Model Definition
# -----------------------------
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.value_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN + PRED_LEN, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.value_emb(x) + self.pos_emb(pos))

class Informer(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=4, e_layers=2, d_layers=1, dropout=0.2):
        super().__init__()
        self.enc_emb = DataEmbedding(input_dim, d_model)
        self.dec_emb = DataEmbedding(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=e_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=d_layers)
        self.proj = nn.Linear(d_model, 1)
    def forward(self, enc_x, dec_y):
        enc_out = self.encoder(self.enc_emb(enc_x))
        dec_out = self.decoder(self.dec_emb(dec_y.unsqueeze(-1)), enc_out)
        return self.proj(dec_out).squeeze(-1)

# -----------------------------
# Training & Evaluation
# -----------------------------
def main():
    # Load data
    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
    if df.isna().any().any():
        print("Found missing values; filling...")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c != 'Electricity_price_MWh']
    print(f"Features: {len(feature_cols)} columns")

    # Split data
    train_df = df.loc['2018-01-01':'2022-12-31']
    val_df = df.loc['2023-01-01':'2023-12-31']
    test_df = df.loc['2024-01-01':'2024-12-31']
    print(f"Train/Val/Test samples: {len(train_df)}, {len(val_df)}, {len(test_df)}")

    # Create datasets and loaders
    train_set = TimeSeriesDataset(train_df, feature_cols, 'Electricity_price_MWh', SEQ_LEN, LABEL_LEN, PRED_LEN)
    val_set = TimeSeriesDataset(val_df, feature_cols, 'Electricity_price_MWh', SEQ_LEN, LABEL_LEN, PRED_LEN)
    test_set = TimeSeriesDataset(test_df, feature_cols, 'Electricity_price_MWh', SEQ_LEN, LABEL_LEN, PRED_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"DataLoaders created (batch size {BATCH_SIZE})")

    # Initialize model
    model = Informer(input_dim=len(feature_cols)).to(DEVICE)
    print("Model initialized on device:", DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    # Training loop
    best_val, wait = float('inf'), 0
    print("Starting training...")
    for epoch in range(1, EPOCHS+1):
        start = time.time()
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training"):  
            enc_x = batch['enc_x'].to(DEVICE)
            dec_y = batch['dec_y'].to(DEVICE)
            optimizer.zero_grad()
            out = model(enc_x, dec_y[:, :LABEL_LEN+PRED_LEN])
            loss = criterion(out[:, -PRED_LEN:], dec_y[:, -PRED_LEN:])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation"):  
                enc_x = batch['enc_x'].to(DEVICE)
                dec_y = batch['dec_y'].to(DEVICE)
                out = model(enc_x, dec_y[:, :LABEL_LEN+PRED_LEN])
                val_loss += criterion(out[:, -PRED_LEN:], dec_y[:, -PRED_LEN:]).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {time.time()-start:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), os.path.join(output_path,'best_informer.pt'))
            print(f"Saved new best model (val {best_val:.4f})")
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"Early stopping invoked after {wait} epochs without improvement.")
                break

    # Testing
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(output_path,'best_informer.pt'), map_location=DEVICE))
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):  
            enc_x = batch['enc_x'].to(DEVICE)
            dec_y = batch['dec_y'].to(DEVICE)
            out = model(enc_x, dec_y[:, :LABEL_LEN+PRED_LEN])
            all_preds.append(out[:, -PRED_LEN:].cpu().numpy())
            all_targs.append(dec_y[:, -PRED_LEN:].cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targs = np.vstack(all_targs)

    mean, var = train_set.scaler.mean_[-1], train_set.scaler.var_[-1]
    std = np.sqrt(var)
    real_preds = all_preds * std + mean
    real_targs = all_targs * std + mean

    # Report metrics
    mse = mean_squared_error(real_targs.flatten(), real_preds.flatten())
    mae = mean_absolute_error(real_targs.flatten(), real_preds.flatten())
    r2 = r2_score(real_targs.flatten(), real_preds.flatten())
    print("Test metrics:")
    print(f"  Real MSE : {mse:.2f}")
    print(f"  Real MAE : {mae:.2f}")
    print(f"  Real R^2 : {r2:.4f}")

    # Plot Actual vs Predicted
    print("Plotting Actual vs Predicted...")
    plt.figure(figsize=(12,6))
    plt.plot(real_targs.flatten(), label='Actual')
    plt.plot(real_preds.flatten(), label='Predicted')
    plt.title('Actual vs Predicted Electricity Prices')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.tight_layout()
    act_path = os.path.join(output_path, f'{mapcode}_actual_vs_predicted.png')
    plt.savefig(act_path)
    print(f"Saved plot to {act_path}")
    plt.show()

    # Permutation Feature Importance
    print("Computing permutation feature importances...")
    baseline = mean_squared_error(real_targs.flatten(), real_preds.flatten())
    importances = []
    orig_feat = test_set.features.copy()
    orig_tar = test_set.targets.copy()
    for idx, feat in enumerate(feature_cols):
        perm_feat = orig_feat.copy()
        np.random.shuffle(perm_feat[:, idx])
        test_set.samples = []
        for i in range(len(perm_feat) - SEQ_LEN - PRED_LEN + 1):
            enc_x = perm_feat[i:i+SEQ_LEN]
            dec_y = orig_tar[i+SEQ_LEN-LABEL_LEN:i+SEQ_LEN+PRED_LEN]
            test_set.samples.append((enc_x, dec_y))
        perm_preds = []
        with torch.no_grad():
            for b in DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False):
                ex = b['enc_x'].to(DEVICE)
                dy = b['dec_y'].to(DEVICE)
                out = model(ex, dy[:, :LABEL_LEN+PRED_LEN])
                perm_preds.append(out[:, -PRED_LEN:].cpu().numpy() * std + mean)
        perm_preds = np.vstack(perm_preds)
        imp_mse = mean_squared_error(real_targs[:len(perm_preds)].flatten(), perm_preds.flatten())
        importances.append((feat, imp_mse - baseline))
    imp_df = pd.DataFrame(importances, columns=['feature','mse_increase']).sort_values('mse_increase', ascending=False)

    # Keep only the top 15
    imp_top15 = imp_df.head(15)

    csv_path = os.path.join(output_path, f'{mapcode}_feature_importances_top15.csv')
    imp_top15.to_csv(csv_path, index=False)
    print(f"Saved top 15 feature importances to {csv_path}")

    # Plot top 15
    print("Plotting top 15 feature importances...")
    top15 = imp_df.head(15)
    plt.figure(figsize=(10,6))
    plt.barh(top15['feature'][::-1], top15['mse_increase'][::-1])
    plt.title('Top 15 Feature Importances (MSE Increase)')
    plt.xlabel('Increase in MSE')
    fi_path = os.path.join(output_path, f'{mapcode}_feature_importance.png')
    plt.savefig(fi_path)
    print(f"Saved feature importance plot to {fi_path}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()