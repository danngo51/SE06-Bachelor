import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pathlib
import joblib

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # Feature projection layer
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # Layer normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # GRU layers
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Determine output dimension size from GRU
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Self-attention mechanism
        self.query = nn.Linear(gru_out_dim, gru_out_dim)
        self.key = nn.Linear(gru_out_dim, gru_out_dim)
        self.value = nn.Linear(gru_out_dim, gru_out_dim)
        self.attention_scale = np.sqrt(gru_out_dim)
        
        # Output projection layers with residual connections
        self.fc1 = nn.Linear(gru_out_dim, gru_out_dim // 2)
        self.act1 = nn.GELU()  # GELU often works better than ReLU variants
        self.ln1 = nn.LayerNorm(gru_out_dim // 2)
        self.fc2 = nn.Linear(gru_out_dim // 2, gru_out_dim // 4)
        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(gru_out_dim // 4)
        self.fc3 = nn.Linear(gru_out_dim // 4, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.feature_proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Process with GRU
        gru_out, hidden = self.gru(x)
        
        # Apply self-attention over the sequence
        if self.bidirectional:
            # Use attention to focus on important parts of the sequence
            q = self.query(gru_out[:, -1, :])  # Use last timestep as query
            k = self.key(gru_out)
            v = self.value(gru_out)
            
            # Calculate attention scores
            attn = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) / self.attention_scale
            attn_weights = F.softmax(attn, dim=2)
            context = torch.bmm(attn_weights, v).squeeze(1)
            
            # Process through output layers with residual connections
            out = self.fc1(context)
            out = self.act1(out)
            out = self.ln1(out)
            out = self.dropout(out)
            
            out = self.fc2(out)
            out = self.act2(out)
            out = self.ln2(out)
            out = self.dropout(out)
            
            return self.fc3(out)
        else:
            # For non-bidirectional, use the last hidden state
            out = self.fc1(hidden[-1])
            out = self.act1(out)
            out = self.ln1(out)
            out = self.dropout(out)
            
            out = self.fc2(out)
            out = self.act2(out)
            out = self.ln2(out)
            out = self.dropout(out)
            
            return self.fc3(out)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class GRUModelTrainer:
    def __init__(self, mapcode="DK1", seq_len=168, pred_len=24, batch_size=64, learning_rate=0.001, 
         num_epochs=50, patience=10):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size  # Larger batch size for faster training
        self.learning_rate = learning_rate  # Higher learning rate
        self.num_epochs = num_epochs  # Fewer epochs
        self.patience = patience  # Less patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 128  # Reduced hidden dimension
        self.num_layers = 2  # Fewer layers
        self.dropout = 0.3  # Add dropout rate
        self.bidirectional = True
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_cols = None
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"
        self.gru_dir.mkdir(exist_ok=True)


    def prepare_sequences(self, df):
        feature_cols = [col for col in df.columns if col not in ['date', 'Electricity_price_MWh']]
        features = df[feature_cols].values
        target = df['Electricity_price_MWh'].values
        
        # Add debug info
        print(f"Feature shape: {features.shape}, Target shape: {target.shape}")
        print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
        
        sequences, targets = [], []
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            sequences.append(features[i:i + self.seq_len])
            targets.append(target[i + self.seq_len:i + self.seq_len + self.pred_len])
        
        seqs_np = np.array(sequences)
        targets_np = np.array(targets)
        
        print(f"Created {len(sequences)} sequences of shape {seqs_np.shape}")
        print(f"Created {len(targets)} targets of shape {targets_np.shape}")
        
        return seqs_np, targets_np, feature_cols

    def train(self, train_file):
        df = pd.read_csv(train_file, parse_dates=['date'], index_col='date').dropna()
        feature_cols = [c for c in df.columns if c != 'Electricity_price_MWh']
        target_col = 'Electricity_price_MWh'
        
        # Handle negative prices before log transform
        min_price = df['Electricity_price_MWh'].min()
        
        # If we have negative or zero values, shift all prices to make them positive
        if min_price <= 0:
            shift_value = abs(min_price) + 1  # Add 1 to ensure all values are positive
            print(f"Shifting electricity prices by {shift_value} to handle negative/zero values")
            df['Electricity_price_MWh'] = df['Electricity_price_MWh'] + shift_value
        
        # Now apply log transform (all values should be positive)
        df['Electricity_price_MWh'] = np.log1p(df['Electricity_price_MWh'])
        
        # Split data with rolling-window validation to maintain time series integrity
        cutoff_val = int(len(df) * 0.8)
        cutoff_test = int(len(df) * 0.9)
        
        train_df = df.iloc[:cutoff_val]
        val_df = df.iloc[cutoff_val:cutoff_test]
        test_df = df.iloc[cutoff_test:]
        
        # Apply data augmentation to training data
        train_sequences, train_targets, self.feature_cols = self.prepare_sequences(train_df)
        val_sequences, val_targets, _ = self.prepare_sequences(val_df)
        test_sequences, test_targets, _ = self.prepare_sequences(test_df)
        
        # Augment training data to improve generalization
        train_sequences, train_targets = self.augment_training_data(train_sequences, train_targets)
        
        # Scale features and targets
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        train_sequences = self.scaler_features.fit_transform(train_sequences.reshape(-1, train_sequences.shape[-1])).reshape(train_sequences.shape)
        train_targets = self.scaler_target.fit_transform(train_targets)
        val_sequences = self.scaler_features.transform(val_sequences.reshape(-1, val_sequences.shape[-1])).reshape(val_sequences.shape)
        val_targets = self.scaler_target.transform(val_targets)
        test_sequences = self.scaler_features.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)
        test_targets = self.scaler_target.transform(test_targets)

        # Create DataLoaders with weighted sampler to handle imbalance
        train_loader = DataLoader(
            TimeSeriesDataset(train_sequences, train_targets), 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Disable multi-worker loading
            pin_memory=False  # Disable pinned memory
        )
        val_loader = DataLoader(TimeSeriesDataset(val_sequences, val_targets), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TimeSeriesDataset(test_sequences, test_targets), batch_size=self.batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function
        self.model = GRUModel(
            train_sequences.shape[-1], 
            self.hidden_dim, 
            self.num_layers, 
            self.pred_len, 
            dropout=0.2,  # Lower dropout
            bidirectional=False  # Disable bidirectional (much faster)
        ).to(self.device)
        
        # Use a combination of L1 and L2 loss for better handling of outliers
        criterion = nn.SmoothL1Loss(beta=0.1)
        
        # Use AdamW with weight decay and gradient clipping
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Simple step learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5
        )

        # Enable logging
        log_file = open(self.gru_dir / "log.txt", "w")
        
        best_val_loss, wait = float('inf'), 0
        for epoch in range(1, self.num_epochs + 1):
            # Training loop
            self.model.train()
            train_loss = 0.0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences, batch_targets = batch_sequences.to(self.device), batch_targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences, batch_targets = batch_sequences.to(self.device), batch_targets.to(self.device)
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Log training and validation loss
            log_message = f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()

            # Early stopping with model checkpoint
            if val_loss < best_val_loss:
                best_val_loss, wait = val_loss, 0
                torch.save(self.model.state_dict(), str(self.gru_dir / "best_gru_model.pth"))
            else:
                wait += 1
                if wait >= self.patience:
                    log_message = "Early stopping triggered."
                    print(log_message)
                    log_file.write(log_message + "\n")
                    break

            # Update the learning rate scheduler
            scheduler.step()  # Only call once per epoch

        # Close log file
        log_file.close()

        # Load the best model
        self.model.load_state_dict(torch.load(str(self.gru_dir / "best_gru_model.pth"), map_location=self.device))

        # Evaluate on test data
        test_loss = 0.0
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                batch_sequences, batch_targets = batch_sequences.to(self.device), batch_targets.to(self.device)
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())
        test_loss /= len(test_loader)

        # Log test loss
        log_file = open(self.gru_dir / "log.txt", "a")
        log_file.write(f"Test Loss: {test_loss:.4f}\n")

        # Rescale predictions and targets and reverse log transform
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_predictions = self.scaler_target.inverse_transform(all_predictions)
        all_targets = self.scaler_target.inverse_transform(all_targets)
        
        # Reverse log transformation and price shifting
        all_predictions = np.expm1(all_predictions)
        all_targets = np.expm1(all_targets)

        # If we shifted the prices before, we need to shift them back
        if min_price <= 0:
            shift_value = abs(min_price) + 1
            all_predictions -= shift_value
            all_targets -= shift_value

        # Calculate metrics
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        
        # Log metrics
        log_message = f"Test metrics: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.close()

        # Save metrics
        metrics_data = {'Metric': ['RMSE', 'MAE', 'R²'], 'Value': [rmse, mae, r2]}
        pd.DataFrame(metrics_data).to_csv(self.gru_dir / "metrics.csv", index=False)
        
        # Save model
        self.save_model()

    def augment_training_data(self, sequences, targets):
        """Minimal augmentation for speed"""
        return sequences, targets  # No augmentation at all for faster training