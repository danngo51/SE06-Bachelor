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

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, bidirectional=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        regressor_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Add attention mechanism
        self.attention = nn.Linear(regressor_input_dim, 1)
        
        # Multi-layer regressor
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, regressor_input_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_input_dim // 2, output_dim)
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        
        # Apply attention over the sequence
        if self.gru.bidirectional:
            # Use the last hidden state from both directions
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            hidden_concat = torch.cat((last_forward, last_backward), dim=1)
            return self.regressor(self.dropout(hidden_concat))
        
        return self.regressor(self.dropout(hidden[-1]))

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class GRUModelTrainer:
    def __init__(self, mapcode="DK1", seq_len=168, pred_len=24, batch_size=32, learning_rate=0.0005, 
             num_epochs=100, patience=15):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size  # Smaller batch size for better generalization
        self.learning_rate = learning_rate  # Lower learning rate
        self.num_epochs = num_epochs  # More epochs with early stopping
        self.patience = patience  # Increased patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 256  # Increased hidden dimension
        self.num_layers = 3  # More layers
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
        sequences, targets = [], []
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            sequences.append(features[i:i + self.seq_len])
            targets.append(target[i + self.seq_len:i + self.seq_len + self.pred_len])
        return np.array(sequences), np.array(targets), feature_cols

    def train(self, train_file):
        df = pd.read_csv(train_file, parse_dates=['date'], index_col='date').dropna()
        feature_cols = [c for c in df.columns if c != 'Electricity_price_MWh']
        target_col = 'Electricity_price_MWh'

        # Split data into train, validation, and test sets
        train_df = df.loc['2018-01-01':'2022-12-31']
        val_df = df.loc['2023-01-01':'2023-12-31']
        test_df = df.loc['2024-01-01':'2024-12-31']

        # Prepare datasets
        train_sequences, train_targets, self.feature_cols = self.prepare_sequences(train_df)
        val_sequences, val_targets, _ = self.prepare_sequences(val_df)
        test_sequences, test_targets, _ = self.prepare_sequences(test_df)

        # Scale features and targets
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        train_sequences = self.scaler_features.fit_transform(train_sequences.reshape(-1, train_sequences.shape[-1])).reshape(train_sequences.shape)
        train_targets = self.scaler_target.fit_transform(train_targets)
        val_sequences = self.scaler_features.transform(val_sequences.reshape(-1, val_sequences.shape[-1])).reshape(val_sequences.shape)
        val_targets = self.scaler_target.transform(val_targets)
        test_sequences = self.scaler_features.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)
        test_targets = self.scaler_target.transform(test_targets)

        # Create DataLoaders
        train_loader = DataLoader(TimeSeriesDataset(train_sequences, train_targets), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(val_sequences, val_targets), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TimeSeriesDataset(test_sequences, test_targets), batch_size=self.batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function
        self.model = GRUModel(
            train_sequences.shape[-1], 
            self.hidden_dim, 
            self.num_layers, 
            self.pred_len, 
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Use Huber loss instead of MSE for robustness to outliers
        criterion = nn.SmoothL1Loss()
        
        # Use AdamW with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        
        # Use cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2
        )
        

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

            # Print training and validation loss
            print(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss, wait = val_loss, 0
                torch.save(self.model.state_dict(), str(self.gru_dir / "best_gru_model.pth"))
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered.")
                    break

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

        # Rescale predictions and targets
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_predictions = self.scaler_target.inverse_transform(all_predictions)
        all_targets = self.scaler_target.inverse_transform(all_targets)

        # Calculate metrics
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        r2 = r2_score(all_targets.flatten(), all_predictions.flatten())

        print(f"Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save metrics
        metrics_data = {'Metric': ['RMSE', 'MAE', 'R²'], 'Value': [rmse, mae, r2]}
        pd.DataFrame(metrics_data).to_csv(self.gru_dir / "metrics.csv", index=False)

    def predict(self, data):
        #data = self.create_features(data)
        sequences, _, _ = self.prepare_sequences(data)
        sequences = self.scaler_features.transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequences_tensor).cpu().numpy()
        return self.scaler_target.inverse_transform(predictions)

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), str(self.gru_dir / "gru_model.pth"))
            joblib.dump(self.scaler_features, str(self.gru_dir / "scaler_features.pkl"))
            joblib.dump(self.scaler_target, str(self.gru_dir / "scaler_target.pkl"))
            joblib.dump({"feature_cols": self.feature_cols}, str(self.gru_dir / "metadata.pkl"))
            print("Model and related files saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        try:
            metadata_path = str(self.gru_dir / "metadata.pkl")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_cols = metadata.get("feature_cols", None)
            else:
                raise FileNotFoundError("Metadata file not found.")

            self.model = GRUModel(len(self.feature_cols), self.hidden_dim, self.num_layers, self.pred_len, self.bidirectional).to(self.device)
            self.model.load_state_dict(torch.load(str(self.gru_dir / "gru_model.pth"), map_location=self.device))
            self.scaler_features = joblib.load(str(self.gru_dir / "scaler_features.pkl"))
            self.scaler_target = joblib.load(str(self.gru_dir / "scaler_target.pkl"))
            print("Model and related files loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def augment_training_data(self, sequences, targets):
        """Apply data augmentation techniques to increase training diversity"""
        aug_sequences = sequences.copy()
        aug_targets = targets.copy()
        
        # Add small Gaussian noise
        noise_factor = 0.05
        noise = noise_factor * np.random.normal(0, 1, aug_sequences.shape)
        aug_sequences += noise
        
        # Combine original and augmented data
        combined_sequences = np.concatenate([sequences, aug_sequences], axis=0)
        combined_targets = np.concatenate([targets, aug_targets], axis=0)
        
        return combined_sequences, combined_targets