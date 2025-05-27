# filepath: d:\Uni\6. semester\MLs\ml_models\Informer\Informer_model.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pathlib
import matplotlib.pyplot as plt
import joblib
try:
    import multiprocessing
    # Ensure spawn method for macOS
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore error
    pass

class DataEmbedding(nn.Module):
    """
    Embedding module for time series data in Informer model.
    Combines value embedding and positional encoding.
    """
    def __init__(self, input_dim, d_model, seq_len=168, pred_len=24):
        super().__init__()
        self.value_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_len + pred_len, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.value_emb(x) + self.pos_emb(pos))


class Informer(nn.Module):
    """
    Informer model for efficient transformer-based time-series forecasting.
    Features lower time complexity and memory usage compared to standard transformers.
    """
    def __init__(self, input_dim, d_model=256, n_heads=4, e_layers=2, d_layers=1, dropout=0.2, 
                seq_len=168, label_len=48, pred_len=24):
        super().__init__()
        self.enc_emb = DataEmbedding(input_dim, d_model, seq_len, pred_len)
        self.dec_emb = DataEmbedding(1, d_model, seq_len, pred_len)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=e_layers)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True),
            num_layers=d_layers)
        
        # Output projection
        self.proj = nn.Linear(d_model, 1)
    
    def forward(self, enc_x, dec_y=None):
        """
        Forward pass for Informer model.
        
        Args:
            enc_x: Encoder input sequence [batch_size, seq_len, input_dim]
            dec_y: Decoder input sequence (optional) [batch_size, label_len+pred_len]
            
        Returns:
            Model output for predictions
        """
        # If dec_y is not provided, create zero placeholder for inference
        if dec_y is None:
            # For inference, we use zeros + historical values from encoder
            # Take the last label_len values from encoder input for decoder start
            dec_y = torch.zeros((enc_x.shape[0], self.label_len+self.pred_len), device=enc_x.device)
            if self.label_len > 0 and enc_x.shape[1] >= self.label_len:
                # If available, use the last values from encoder input
                target_col_idx = -1  # Assuming target is last column in features
                last_values = enc_x[:, -self.label_len:, target_col_idx]
                dec_y[:, :self.label_len] = last_values
        
        # Encoder
        enc_out = self.encoder(self.enc_emb(enc_x))
        
        # Decoder
        dec_out = self.decoder(self.dec_emb(dec_y.unsqueeze(-1)), enc_out)
        
        # Project to output
        return self.proj(dec_out).squeeze(-1)


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data used in Informer model training.
    Handles feature scaling and sequence preparation.
    """
    def __init__(self, df, feature_cols, target_col, seq_len, label_len, pred_len, scaler=None):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Extract data
        data = df[feature_cols + [target_col]].values.astype(np.float32)
        
        # Scale data
        if scaler is None:
            self.scaler = StandardScaler().fit(data)
        else:
            self.scaler = scaler
            
        data = self.scaler.transform(data)
        
        # Separate features and target
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        
        # Create samples
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


class InformerModelTrainer:
    """
    Trainer class for Informer model.
    Handles training, evaluation, and model persistence.
    """
    def __init__(self, mapcode="DK1", seq_len=168, label_len=48, pred_len=24, 
                 batch_size=32, learning_rate=1e-4, epochs=20, 
                 early_stop_patience=5, device=None):
        """
        Initialize the Informer model trainer.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
            seq_len: Length of input sequence
            label_len: Length of label sequence (overlap with pred_len)
            pred_len: Length of prediction sequence
            batch_size: Training batch size
            learning_rate: Model learning rate
            epochs: Maximum training epochs
            early_stop_patience: Patience for early stopping
            device: Torch device (auto-detect if None)
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                                      "cpu")
        else:
            self.device = device
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.informer_dir = self.data_dir / "informer"
        
        # Create output directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.informer_dir, exist_ok=True)
    
    def train(self, data_path=None):
        """
        Train the Informer model on the provided data.
        
        Args:
            data_path: Path to the training data CSV file (optional)
            
        Returns:
            Dictionary with training results and metrics
        """
        # Print all settings at the start
        print("=" * 60)
        print("INFORMER MODEL TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Market Code (mapcode): {self.mapcode}")
        print(f"Sequence Length (seq_len): {self.seq_len}")
        print(f"Label Length (label_len): {self.label_len}")
        print(f"Prediction Length (pred_len): {self.pred_len}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Maximum Epochs: {self.epochs}")
        print(f"Early Stop Patience: {self.early_stop_patience}")
        print(f"Device: {self.device}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Model Directory: {self.informer_dir}")
        print("=" * 60)
        
        # Use default path if not provided
        if data_path is None:
            data_path = self.data_dir / f"{self.mapcode}_full_data_2018_2024.csv"
          # Load and prepare data
        print(f"\n📁 Loading data from: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        print(f"📊 Original data shape: {df.shape}")
        
        # Handle missing values
        if df.isna().any().any():
            print("⚠️  Found missing values; filling...")
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.dropna(inplace=True)
            print(f"📊 Data shape after cleaning: {df.shape}")
        
        # Define features and target
        feature_cols = [c for c in df.columns if c != 'Electricity_price_MWh']
        target_col = 'Electricity_price_MWh'
        print(f"🔢 Features: {len(feature_cols)} columns")
        print(f"🎯 Target: {target_col}")
        
        # Split data
        train_df = df.loc['2018-01-01':'2022-12-31']
        val_df = df.loc['2023-01-01':'2023-12-31']
        test_df = df.loc['2024-01-01':'2024-12-31']
        print(f"📈 Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"📅 Train period: {train_df.index.min()} to {train_df.index.max()}")
        print(f"📅 Val period: {val_df.index.min()} to {val_df.index.max()}")
        print(f"📅 Test period: {test_df.index.min()} to {test_df.index.max()}")
        
        # Create datasets and loaders
        train_set = TimeSeriesDataset(train_df, feature_cols, target_col, 
                                    self.seq_len, self.label_len, self.pred_len)
        val_set = TimeSeriesDataset(val_df, feature_cols, target_col, 
                                  self.seq_len, self.label_len, self.pred_len, 
                                  scaler=train_set.scaler)
        test_set = TimeSeriesDataset(test_df, feature_cols, target_col, 
                                   self.seq_len, self.label_len, self.pred_len,
                                   scaler=train_set.scaler)
          # Save scaler for later use
        joblib.dump(train_set.scaler, self.informer_dir / "scaler.pkl")
        print(f"Saved scaler to {self.informer_dir / 'scaler.pkl'}")
        
        # Save feature columns for later use during prediction
        joblib.dump(feature_cols, self.informer_dir / "feature_columns.pkl")
        print(f"Saved feature columns to {self.informer_dir / 'feature_columns.pkl'}")
        
        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        print(f"DataLoaders created (batch size {self.batch_size})")
        
        # Initialize model
        model = Informer(input_dim=len(feature_cols), seq_len=self.seq_len, 
                        label_len=self.label_len, pred_len=self.pred_len).to(self.device)
        print(f"Model initialized on device: {self.device}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        
        # Training loop
        best_val, wait = float('inf'), 0
        train_losses, val_losses = [], []
        print("Starting training...")
        
        for epoch in range(1, self.epochs+1):
            start_time = pd.Timestamp.now()
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} - Training"):  
                enc_x = batch['enc_x'].to(self.device)
                dec_y = batch['dec_y'].to(self.device)
                
                optimizer.zero_grad()
                out = model(enc_x, dec_y[:, :self.label_len+self.pred_len])
                loss = criterion(out[:, -self.pred_len:], dec_y[:, -self.pred_len:])
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{self.epochs} - Validation"):  
                    enc_x = batch['enc_x'].to(self.device)
                    dec_y = batch['dec_y'].to(self.device)
                    
                    out = model(enc_x, dec_y[:, :self.label_len+self.pred_len])
                    val_loss += criterion(out[:, -self.pred_len:], dec_y[:, -self.pred_len:]).item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
              # Print progress
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            gap_ratio = val_loss / train_loss if train_loss > 0 else 1.0
            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Gap Ratio: {gap_ratio:.2f} | Time: {duration:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
            
            # Check for best model with overfitting protection
            overfitting_threshold = 2.0  # Val loss should not be more than 2x train loss
            is_overfitting = gap_ratio > overfitting_threshold
            
            if val_loss < best_val:
                if not is_overfitting:
                    best_val, wait = val_loss, 0
                    torch.save(model.state_dict(), self.informer_dir / 'best_informer.pt')
                    torch.save(model, self.informer_dir / 'informer_model.pt')
                    print(f"✅ Saved new best model (val loss: {best_val:.4f})")
                else:
                    wait += 1
                    print(f"⚠️  Skipped saving - potential overfitting detected (gap ratio: {gap_ratio:.2f})")
                    print(f"⏳ Early stopper: {wait} of {self.early_stop_patience}")
            else:
                wait += 1
                print(f"⏳ Early stopper: {wait} of {self.early_stop_patience}")
                
            if wait >= self.early_stop_patience:
                print(f"🛑 Early stopping triggered after {wait} epochs without improvement.")
                break
        
        # Testing
        print("Evaluating on test set...")
        model.load_state_dict(torch.load(self.informer_dir / 'best_informer.pt', map_location=self.device))
        model.eval()
        
        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):  
                enc_x = batch['enc_x'].to(self.device)
                dec_y = batch['dec_y'].to(self.device)
                
                out = model(enc_x, dec_y[:, :self.label_len+self.pred_len])
                all_preds.append(out[:, -self.pred_len:].cpu().numpy())
                all_targs.append(dec_y[:, -self.pred_len:].cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targs = np.vstack(all_targs)

        # Inverse transform predictions and targets
        mean, var = train_set.scaler.mean_[-1], train_set.scaler.var_[-1]
        std = np.sqrt(var)
        real_preds = all_preds * std + mean
        real_targs = all_targs * std + mean

        # Calculate metrics
        mse = mean_squared_error(real_targs.flatten(), real_preds.flatten())
        mae = mean_absolute_error(real_targs.flatten(), real_preds.flatten())
        r2 = r2_score(real_targs.flatten(), real_preds.flatten())
        
        print("Test metrics:")
        print(f"  MSE : {mse:.2f}")
        print(f"  MAE : {mae:.2f}")
        print(f"  R^2 : {r2:.4f}")

        # Plot and save results
        self._plot_results(real_targs.flatten(), real_preds.flatten(), train_losses, val_losses)          # Save model metadata
        metadata = {
            "mapcode": self.mapcode,
            "seq_len": self.seq_len,
            "label_len": self.label_len,
            "pred_len": self.pred_len,
            "training_settings": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "early_stop_patience": self.early_stop_patience,
                "device": str(self.device),
                "optimizer": "Adam",
                "weight_decay": 1e-4,
                "scheduler": "ReduceLROnPlateau",
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "gradient_clipping": 1.0,
                "overfitting_threshold": overfitting_threshold
            },
            "training_losses": {
                "final_train_loss": float(train_losses[-1]) if train_losses else None,
                "final_val_loss": float(val_losses[-1]) if val_losses else None,
                "best_val_loss": float(best_val),
                "total_epochs": len(train_losses)
            },
            "metrics": {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "feature_count": len(feature_cols),
            "features": feature_cols
        }
        
        # Save in a format that can be easily loaded
        with open(self.informer_dir / "model_metadata.txt", "w") as f:
            f.write(str(metadata))
        
        return {
            "model_path": str(self.informer_dir / 'informer_model.pt'),
            "scaler_path": str(self.informer_dir / 'scaler.pkl'),
            "metrics": {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            }
        }
    
    def _plot_results(self, true_values, predictions, train_losses, val_losses):
        """
        Plot and save model results.
        
        Args:
            true_values: Actual values from test set
            predictions: Model predictions
            train_losses: Training loss history
            val_losses: Validation loss history
        """
        # Plot 1: Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(true_values[:min(336, len(true_values))], label='Actual')  # Show up to 2 weeks
        plt.plot(predictions[:min(336, len(predictions))], label='Predicted')
        plt.title(f'Actual vs Predicted Electricity Prices - {self.mapcode}')
        plt.xlabel('Test Sample Index (Hours)')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.informer_dir / f'{self.mapcode}_actual_vs_predicted.png')
        
        # Plot 2: Training and Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Loss - {self.mapcode}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.informer_dir / f'{self.mapcode}_training_validation_loss.png')
        
        # Close plots to free memory
        plt.close('all')


# Function to run training if script is executed directly
def train_informer_model(mapcode="DK1"):
    """
    Train an Informer model for the specified market area.
    
    Args:
        mapcode: String code for the market area (e.g., "DK1")
        
    Returns:
        Dictionary with training results
    """
    trainer = InformerModelTrainer(mapcode=mapcode)
    return trainer.train()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        train_informer_model(mapcode=sys.argv[1])
    else:
        train_informer_model()
