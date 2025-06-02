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
from tqdm import tqdm

class GRU(nn.Module):
    """
    GRU model for electricity price forecasting with simplified architecture
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Simple GRU architecture
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: [batch_size, seq_len, input_dim]
        
        # Run GRU
        output, h_n = self.gru(x)
        
        # Take the last hidden state
        last_hidden = h_n[-1, :, :]  # shape: [batch_size, hidden_dim]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Output layer
        out = self.fc(last_hidden)  # shape: [batch_size, output_dim]
        
        return out


class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GRUTrainer:
    """Trainer class for GRU model"""
    def __init__(self, mapcode="DK1", seq_len=168, pred_len=24, 
                 hidden_dim=128, num_layers=2, dropout=0.2, 
                 batch_size=32, learning_rate=0.001, num_epochs=50, patience=10):
        """
        Initialize GRU trainer
        
        Args:
            mapcode: Region code (DK1, DK2, etc.)
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of output prediction (forecast horizon)
            hidden_dim: Hidden dimension of GRU
            num_layers: Number of GRU layers
            dropout: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.model_dir = self.data_dir / "gru"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model and training attributes
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_cols = None
        self.price_shift = 0
    
    def create_sequences(self, features, targets, seq_len, pred_len):
        """Create sequences for training"""
        X, y = [], []
        for i in range(len(features) - seq_len - pred_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)
    
    def train(self, train_file):
        """Train the GRU model"""
        print(f"Starting GRU training for {self.mapcode}...")
        
        # Load and prepare data
        df = pd.read_csv(train_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"Data ranges from {df['date'].min()} to {df['date'].max()}")
        
        # Handle price data
        target_col = 'Electricity_price_MWh'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Log and store min price for shifting
        min_price = df[target_col].min()
        if min_price <= 0:
            # Shift prices to make them positive for log transform
            self.price_shift = abs(min_price) + 1
            print(f"Shifting prices by {self.price_shift} to make them positive")
            df[target_col] = df[target_col] + self.price_shift
        
        # Apply log transform to handle price spikes
        df[target_col + '_log'] = np.log1p(df[target_col])
        
        # Split data: train (80%), validation (10%), test (10%)
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
        
        # Determine feature columns (exclude date and target)
        exclude_cols = ['date', target_col, target_col + '_log']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(self.feature_cols)} features")
        
        # Create sequences
        X_train = train_df[self.feature_cols].values
        y_train = train_df[target_col + '_log'].values
        X_val = val_df[self.feature_cols].values
        y_val = val_df[target_col + '_log'].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df[target_col + '_log'].values
        
        # Scale features
        self.scaler_features = StandardScaler()
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_val_scaled = self.scaler_features.transform(X_val)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Scale targets
        self.scaler_target = StandardScaler()
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled, self.seq_len, self.pred_len)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled, self.seq_len, self.pred_len)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled, self.seq_len, self.pred_len)
        
        print(f"Training sequences: {X_train_seq.shape}, {y_train_seq.shape}")
        print(f"Validation sequences: {X_val_seq.shape}, {y_val_seq.shape}")
        print(f"Test sequences: {X_test_seq.shape}, {y_test_seq.shape}")
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
        test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model = ElectricityGRU(
            input_dim=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.pred_len,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Start training log
        log_file = open(self.model_dir / "training_log.txt", "w")
        log_file.write(f"Training GRU model for {self.mapcode}\n")
        log_file.write(f"Sequence length: {self.seq_len}, Prediction length: {self.pred_len}\n")
        log_file.write(f"Hidden dimension: {self.hidden_dim}, Layers: {self.num_layers}\n")
        log_file.write(f"Features: {len(self.feature_cols)}\n\n")
        
        for epoch in range(1, self.num_epochs + 1):
            # Training
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
            
            # Calculate average losses
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            log_message = f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), self.model_dir / "best_model.pth")
                print(f"Model saved at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    log_file.write(f"Early stopping at epoch {epoch}\n")
                    break
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load(self.model_dir / "best_model.pth"))
        
        # Evaluate on test set
        self.model.eval()
        test_predictions = []
        test_actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                test_predictions.append(outputs.cpu().numpy())
                test_actuals.append(y_batch.numpy())
        
        # Convert to arrays
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_actuals = np.concatenate(test_actuals, axis=0)
        
        # Inverse transform predictions and actuals
        test_predictions = test_predictions.reshape(-1, 1)
        test_actuals = test_actuals.reshape(-1, 1)
        
        test_predictions = self.scaler_target.inverse_transform(test_predictions).flatten()
        test_actuals = self.scaler_target.inverse_transform(test_actuals).flatten()
        
        # Inverse log transform
        test_predictions = np.expm1(test_predictions)
        test_actuals = np.expm1(test_actuals)
        
        # Shift back to original scale if needed
        if self.price_shift > 0:
            test_predictions -= self.price_shift
            test_actuals -= self.price_shift
        
        # Calculate metrics
        mse = mean_squared_error(test_actuals, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_actuals, test_predictions)
        r2 = r2_score(test_actuals, test_predictions)
        
        # Log metrics
        log_message = f"\nTest Metrics:\n"
        log_message += f"MSE: {mse:.4f}\n"
        log_message += f"RMSE: {rmse:.4f}\n"
        log_message += f"MAE: {mae:.4f}\n"
        log_message += f"R²: {r2:.4f}\n"
        
        print(log_message)
        log_file.write(log_message)
        log_file.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [mse, rmse, mae, r2]
        })
        metrics_df.to_csv(self.model_dir / "metrics.csv", index=False)
        
        # Save model and metadata
        self.save_model()
        
        return self.model
    
    def save_model(self):
        """Save the model and related metadata"""
        # Create directory if not exists
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = self.model_dir / "gru_model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save feature scaler
        joblib.dump(self.scaler_features, self.model_dir / "feature_scaler.pkl")
        
        # Save target scaler
        joblib.dump(self.scaler_target, self.model_dir / "target_scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_cols, self.model_dir / "feature_columns.pkl")
        
        # Save model configuration
        model_config = {
            'input_dim': len(self.feature_cols),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.pred_len,
            'dropout': self.dropout,
            'price_shift': self.price_shift
        }
        joblib.dump(model_config, self.model_dir / "model_config.pkl")
        
        print(f"Model and metadata saved to {self.model_dir}")