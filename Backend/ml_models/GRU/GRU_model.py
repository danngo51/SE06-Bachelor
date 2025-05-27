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
import warnings

# Suppress matplotlib font warnings for Unicode characters (emojis)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")


class GRUModel(nn.Module):
    """
    GRU neural network model for electricity price forecasting.
    """
    
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, output_dim=24, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # If bidirectional, we need to double the size of the hidden dimension for the regressor
        regressor_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.regressor = nn.Linear(regressor_input_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → e.g. [32, 168, 512]
        _, hidden = self.gru(x)  # hidden: [num_layers * (2 if bidir else 1), batch, hidden_dim]
        
        # If bidirectional, concatenate the last forward and backward hidden states
        if self.bidirectional:
            # For bidirectional, hidden shape: [num_layers*2, batch, hidden_dim]
            # Extract the last forward and backward hidden states
            last_forward = hidden[-2]  # Second-to-last layer is the forward of the last layer
            last_backward = hidden[-1]  # Last layer is the backward of the last layer
            hidden_concat = torch.cat((last_forward, last_backward), dim=1)
            output = self.regressor(hidden_concat)
        else:
            # For unidirectional, hidden shape: [num_layers, batch, hidden_dim]
            output = self.regressor(hidden[-1])  # use last layer's hidden state
            
        return output  # e.g. [batch, 24] → next 24h price prediction

class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data used by GRU model.
    """
    
    def __init__(self, sequences, targets):
        """
        Initialize the dataset.
        
        Args:
            sequences: Input sequences [n_samples, seq_len, n_features]
            targets: Target values [n_samples, pred_len]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GRUModelTrainer:
    """
    Consolidated GRU model for electricity price forecasting.
    Handles training, validation, and prediction using GRU architecture.
    """
    
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, pred_len: int = 24):
        """
        Initialize the GRU model trainer.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
            seq_len: Length of input sequence (default: 168 hours = 1 week)
            pred_len: Length of prediction sequence (default: 24 hours = 1 day)
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.input_dim = None  # Will be set after preprocessing
        self.hidden_dim = 128
        self.num_layers = 2
        self.bidirectional = True
        self.dropout = 0.2
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.patience = 10
        
        # Model and scalers
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"
        self.gru_dir.mkdir(exist_ok=True)
        
        print(f"GRU Model initialized for {mapcode}")
        print(f"Device: {self.device}")
        print(f"Data directory: {self.data_dir}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the GRU model.
        
        Args:
            df: Input dataframe with basic features
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['date', 'hour', 'Electricity_price_MWh']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe")
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Price-based features (lagged values and rolling statistics)
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24, 48, 168]:
            df[f'price_roll_mean_{window}'] = df['Electricity_price_MWh'].shift(1).rolling(window, min_periods=1).mean()
            df[f'price_roll_std_{window}'] = df['Electricity_price_MWh'].shift(1).rolling(window, min_periods=1).std()
        
        # Load-related features if available
        if 'DAHTL_TotalLoadValue' in df.columns:
            for lag in [1, 24, 168]:
                df[f'load_lag_{lag}'] = df['DAHTL_TotalLoadValue'].shift(lag)
            
            for window in [6, 24, 168]:
                df[f'load_roll_mean_{window}'] = df['DAHTL_TotalLoadValue'].shift(1).rolling(window, min_periods=1).mean()
        
        # Renewable energy features if available
        renewable_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['wind', 'solar', 'renewable'])]
        for col in renewable_cols:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_roll_mean_24'] = df[col].shift(1).rolling(24, min_periods=1).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """
        Prepare sequences for training/prediction.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Tuple of (sequences, targets, feature_columns)
        """
        # Define feature columns (exclude target and date/time columns)
        exclude_cols = ['date', 'Electricity_price_MWh']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get features and target
        features = df[feature_cols].values
        target = df['Electricity_price_MWh'].values
        
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            # Input sequence
            seq = features[i:i + self.seq_len]
            # Target sequence (next pred_len hours)
            tgt = target[i + self.seq_len:i + self.seq_len + self.pred_len]
            
            sequences.append(seq)
            targets.append(tgt)
        
        return np.array(sequences), np.array(targets), feature_cols
    
    def train(self, train_file: str, val_file: str = None, save_model: bool = True):
        """
        Train the GRU model.
        
        Args:
            train_file: Path to training data CSV file
            val_file: Path to validation data CSV file (optional)
            save_model: Whether to save the trained model
        """
        print("Loading and preprocessing training data...")
        
        # Load training data
        train_df = pd.read_csv(train_file, parse_dates=['date'])
        train_df = self.create_features(train_df)
        
        # Prepare training sequences
        train_sequences, train_targets, feature_cols = self.prepare_sequences(train_df)
        self.input_dim = train_sequences.shape[-1]
        
        print(f"Training data shape: {train_sequences.shape}")
        print(f"Target data shape: {train_targets.shape}")
        print(f"Number of features: {self.input_dim}")
        
        # Scale features and targets
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        
        # Reshape for scaling
        train_sequences_reshaped = train_sequences.reshape(-1, self.input_dim)
        train_sequences_scaled = self.scaler_features.fit_transform(train_sequences_reshaped)
        train_sequences_scaled = train_sequences_scaled.reshape(train_sequences.shape)
        
        train_targets_scaled = self.scaler_target.fit_transform(train_targets)
        
        # Prepare validation data if provided
        val_sequences_scaled = None
        val_targets_scaled = None
        if val_file and os.path.exists(val_file):
            print("Loading and preprocessing validation data...")
            val_df = pd.read_csv(val_file, parse_dates=['date'])
            val_df = self.create_features(val_df)
            val_sequences, val_targets, _ = self.prepare_sequences(val_df)
            
            # Scale validation data
            val_sequences_reshaped = val_sequences.reshape(-1, self.input_dim)
            val_sequences_scaled = self.scaler_features.transform(val_sequences_reshaped)
            val_sequences_scaled = val_sequences_scaled.reshape(val_sequences.shape)
            val_targets_scaled = self.scaler_target.transform(val_targets)
        
        # Create datasets and data loaders
        train_dataset = TimeSeriesDataset(train_sequences_scaled, train_targets_scaled)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if val_sequences_scaled is not None:
            val_dataset = TimeSeriesDataset(val_sequences_scaled, val_targets_scaled)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = GRUModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.pred_len,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)        # Training loop with enhanced visualization
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        print("=" * 60)
        print("GRU MODEL TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Market Code (mapcode): {self.mapcode}")
        print(f"Sequence Length: {self.seq_len}")
        print(f"Prediction Length: {self.pred_len}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Bidirectional: {self.bidirectional}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Maximum Epochs: {self.num_epochs}")
        print(f"Early Stop Patience: {self.patience}")
        print(f"Device: {self.device}")
        print("=" * 60)
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            start_time = pd.Timestamp.now()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_sequences, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Training"):
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_sequences, batch_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation"):
                        batch_sequences = batch_sequences.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = self.model(batch_sequences)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                # Calculate training duration and gap ratio
                duration = (pd.Timestamp.now() - start_time).total_seconds()
                gap_ratio = val_loss / train_loss if train_loss > 0 else 1.0
                  # Print enhanced progress information
                print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Gap Ratio: {gap_ratio:.2f} | Time: {duration:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
                
                # Enhanced early stopping with overfitting protection
                overfitting_threshold = 1.1  # Val loss should not be more than 2x train loss
                is_overfitting = gap_ratio > overfitting_threshold
                
                if val_loss < best_val_loss:
                    # Always update best loss and reset patience when we find improvement
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_model:
                        self._save_model()
                    
                    if not is_overfitting:
                        print(f"✅ Saved new best model (val loss: {best_val_loss:.4f})")
                    else:
                        print(f"✅ Saved new best model (val loss: {best_val_loss:.4f}) - ⚠️ Warning: potential overfitting (gap ratio: {gap_ratio:.2f})")
                
                else:
                    patience_counter += 1
                    if is_overfitting:
                        print(f"⚠️ No improvement and potential overfitting detected (gap ratio: {gap_ratio:.2f})")
                    print(f"⏳ Early stopper: {patience_counter} of {self.patience}")
                
                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
                    break
            else:
                # Training only (no validation) - track best training loss with early stopping
                duration = (pd.Timestamp.now() - start_time).total_seconds()
                print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Time: {duration:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
                
                # Early stopping based on training loss improvement
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience_counter = 0  # Reset patience counter on improvement
                    if save_model:
                        self._save_model()
                    print(f"✅ Saved new best model (train loss: {best_train_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"⏳ Early stopper: {patience_counter} of {self.patience}")
                
                # Check for early stopping
                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
                    break
                
                # For training-only, save periodically as well
                if save_model and (epoch + 1) % 10 == 0:
                    print(f"📁 Periodic save at epoch {epoch+1}")
          # Final model save
        if save_model:
            if val_loader:
                print(f"🏆 Training completed! Best validation loss: {best_val_loss:.4f}")
            else:
                print(f"🏆 Training completed! Best training loss: {best_train_loss:.4f}")
                self._save_model()  # Ensure final save for training-only mode
        
        print("🎯 Training completed!")
        
        # Generate enhanced training visualizations
        self._plot_training_results(train_losses, val_losses, best_train_loss, best_val_loss)
        
        return train_losses, val_losses
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input dataframe
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        # Preprocess data
        data = self.create_features(data)
        sequences, _, _ = self.prepare_sequences(data)
        
        # Scale sequences
        sequences_reshaped = sequences.reshape(-1, self.input_dim)
        sequences_scaled = self.scaler_features.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(sequences_tensor).cpu().numpy()
        
        # Inverse scale predictions
        predictions = self.scaler_target.inverse_transform(predictions_scaled)
        
        return predictions
    
    def _save_model(self):
        """Save the trained model and scalers."""
        # Save model state dict
        torch.save(self.model.state_dict(), str(self.gru_dir / "gru_model.pth"))
        
        # Save entire model for easier loading
        torch.save(self.model, str(self.gru_dir / "gru_model_complete.pth"))
        
        # Save scalers
        joblib.dump(self.scaler_features, str(self.gru_dir / "scaler_features.pkl"))
        joblib.dump(self.scaler_target, str(self.gru_dir / "scaler_target.pkl"))
        
        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'pred_len': self.pred_len,
            'seq_len': self.seq_len,
            'bidirectional': self.bidirectional,
            'mapcode': self.mapcode
        }
        joblib.dump(config, str(self.gru_dir / "model_config.pkl"))
        
        print(f"Model saved to {self.gru_dir}")
    
    def load_model(self, model_dir: str = None):
        """
        Load a pre-trained model.
        
        Args:
            model_dir: Directory containing the saved model (optional)
        """
        if model_dir:
            model_path = pathlib.Path(model_dir)
        else:
            model_path = self.gru_dir
        
        try:
            # Load model configuration
            config = joblib.load(str(model_path / "model_config.pkl"))
            self.input_dim = config['input_dim']
            self.hidden_dim = config['hidden_dim']
            self.num_layers = config['num_layers']
            self.pred_len = config['pred_len']
            self.seq_len = config['seq_len']
            self.bidirectional = config['bidirectional']
            
            # Load complete model
            self.model = torch.load(str(model_path / "gru_model_complete.pth"), map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
              # Load scalers
            self.scaler_features = joblib.load(str(model_path / "scaler_features.pkl"))
            self.scaler_target = joblib.load(str(model_path / "scaler_target.pkl"))            
            print(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _plot_training_results(self, train_losses: list, val_losses: list = None, best_train_loss: float = None, best_val_loss: float = None):
        """
        Plot enhanced training results including loss curves and test predictions.
        
        Args:
            train_losses: Training loss history
            val_losses: Validation loss history (optional)
            best_train_loss: Best training loss achieved (optional)
            best_val_loss: Best validation loss achieved (optional)
        """
        print("📊 Generating training visualizations...")
        
        # Plot 1: Training and Validation Loss
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        plt.title(f'GRU Model Training History - {self.mapcode}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
          # Save training history plot
        loss_plot_path = self.gru_dir / f"{self.mapcode}_training_validation_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training loss plot to {loss_plot_path}")
        
        # Show plot
        plt.show()
        plt.close()
          # Plot 2: Training Summary Information
        plt.figure(figsize=(10, 6))
          # Create text summary
        summary_text = f"""
GRU MODEL TRAINING SUMMARY - {self.mapcode}

Training Configuration:
• Sequence Length: {self.seq_len} hours
• Prediction Length: {self.pred_len} hours  
• Hidden Dimension: {self.hidden_dim}
• Number of Layers: {self.num_layers}
• Bidirectional: {self.bidirectional}
• Batch Size: {self.batch_size}
• Learning Rate: {self.learning_rate}
• Device: {self.device}

Training Results:
• Total Epochs: {len(train_losses)}
• Best Training Loss: {best_train_loss:.4f if best_train_loss is not None else min(train_losses):.4f}
• Best Validation Loss: {best_val_loss:.4f if best_val_loss is not None and best_val_loss != float('inf') else 'N/A'}
• Model Saved: YES
"""
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        plt.axis('off')
        plt.tight_layout()
        
        # Save summary plot
        summary_plot_path = self.gru_dir / f"{self.mapcode}_training_summary.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training summary to {summary_plot_path}")
        
        plt.show()
        plt.close()
        
        print("📈 Training visualization completed!")

    def _plot_test_results(self, true_values: np.ndarray, predictions: np.ndarray):
        """
        Plot test results comparing actual vs predicted values.
        
        Args:
            true_values: Actual test values
            predictions: Model predictions
        """
        print("📊 Generating test results visualization...")
        
        # Flatten arrays if needed
        if len(true_values.shape) > 1:
            true_values = true_values.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Limit to reasonable display size (first 2 weeks = 336 hours)
        display_length = min(336, len(true_values), len(predictions))
        true_display = true_values[:display_length]
        pred_display = predictions[:display_length]
        
        # Plot actual vs predicted
        plt.figure(figsize=(15, 8))
        
        # Main comparison plot
        plt.subplot(2, 1, 1)
        plt.plot(true_display, label='Actual', color='blue', linewidth=2, alpha=0.8)
        plt.plot(pred_display, label='Predicted', color='red', linewidth=2, alpha=0.8)
        plt.title(f'GRU Model: Actual vs Predicted Electricity Prices - {self.mapcode}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Test Sample Index (Hours)', fontsize=12)
        plt.ylabel('Price (EUR/MWh)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Scatter plot for correlation
        plt.subplot(2, 1, 2)
        plt.scatter(true_display, pred_display, alpha=0.6, color='green', s=20)
        
        # Perfect prediction line
        min_val = min(true_display.min(), pred_display.min())
        max_val = max(true_display.max(), pred_display.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price (EUR/MWh)', fontsize=12)
        plt.ylabel('Predicted Price (EUR/MWh)', fontsize=12)
        plt.title('Prediction Correlation', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        test_plot_path = self.gru_dir / f"{self.mapcode}_actual_vs_predicted.png"
        plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved test results plot to {test_plot_path}")
        
        plt.show()
        plt.close()
        
        print("📈 Test visualization completed!")
    
    def evaluate(self, test_file: str) -> dict:
        """
        Evaluate the model on test data with enhanced visualization.
        
        Args:
            test_file: Path to test data CSV file
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        print("📊 Starting model evaluation...")
        
        # Load and preprocess test data
        test_df = pd.read_csv(test_file, parse_dates=['date'])
        test_df = self.create_features(test_df)
        
        # Make predictions
        predictions = self.predict(test_df)
        
        # Get actual values (last pred_len values for each sequence)
        actual_values = []
        for i in range(len(test_df) - self.seq_len - self.pred_len + 1):
            actual = test_df['Electricity_price_MWh'].iloc[i + self.seq_len:i + self.seq_len + self.pred_len].values
            actual_values.append(actual)
        
        actual_values = np.array(actual_values)
        
        # Calculate metrics
        mse = mean_squared_error(actual_values.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values.flatten(), predictions.flatten())
        r2 = r2_score(actual_values.flatten(), predictions.flatten())
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual_values.flatten() - predictions.flatten()) / actual_values.flatten())) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("📈 Evaluation Metrics:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"  {metric:>6}: {value:.4f}")
        print("=" * 40)
        
        # Generate test visualization
        self._plot_test_results(actual_values, predictions)
        
        return metrics
