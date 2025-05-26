import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class XGBoostRegimeModel:
    """
    XGBoost model with regime-switching capability for electricity price forecasting.
    Uses separate models for normal and spike price regimes based on lagged price values.
    
    This is the consolidated implementation that uses the regime-based approach,
    which provides better performance by handling price spikes separately.
    """
    
    def __init__(self, mapcode="DK1", threshold_percentile=0.90):
        """
        Initialize the regime-based XGBoost model.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
            threshold_percentile: Percentile threshold for defining price spikes (default: 0.90)
        """
        self.mapcode = mapcode
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.model_normal = None
        self.model_spike = None
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.regime_dir = self.data_dir / "regime_models"
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.regime_dir, exist_ok=True)
    
    def _engineer_features(self, df):
        """
        Add engineered features to the dataframe.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with added features
        """
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Rolling statistics
        df['price_roll_3h'] = df['Electricity_price_MWh'].shift(1).rolling(3, min_periods=1).mean()
        df['load_roll_6h'] = df['DAHTL_TotalLoadValue'].shift(1).rolling(6, min_periods=1).mean()
        
        # Interaction term
        df['gas_x_load'] = df['Natural_Gas_price_EUR'] * df['DAHTL_TotalLoadValue']
        
        # One-hour lag (needed for regime split)
        df['EP_lag_1'] = df['Electricity_price_MWh'].shift(1)
        
        # Drop rows with NaN values
        return df.dropna()
    
    def train(self, training_file, xgb_params=None):
        """
        Train separate models for normal and spike regimes.
        
        Args:
            training_file: CSV file with training data
            xgb_params: Parameters for XGBoost model (optional)
        """
        # Default XGBoost parameters if none provided
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'random_state': 42
            }
        
        # Load and preprocess training data
        df = pd.read_csv(os.path.join(self.data_dir, training_file), parse_dates=['date'])
        df = self._engineer_features(df)
        
        # Define regime threshold
        self.threshold = df['Electricity_price_MWh'].quantile(self.threshold_percentile)
        print(f"Threshold for spike regime: {self.threshold:.2f} €/MWh")
        
        # Split data by regime
        mask_spike = df['EP_lag_1'] > self.threshold
        mask_normal = ~mask_spike
        
        df_normal = df[mask_normal].copy()
        df_spike = df[mask_spike].copy()
        
        print(f"Normal regime: {len(df_normal)} samples ({len(df_normal)/len(df)*100:.1f}%)")
        print(f"Spike regime: {len(df_spike)} samples ({len(df_spike)/len(df)*100:.1f}%)")
        
        # Prepare features and targets for each regime
        features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        
        X_norm = df_normal[features]
        y_norm = df_normal['Electricity_price_MWh']
        X_spk = df_spike[features]
        y_spk = df_spike['Electricity_price_MWh']
        
        # Split into train/test for each regime
        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_norm, y_norm, test_size=0.2, shuffle=False)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_spk, y_spk, test_size=0.2, shuffle=False)
        
        # Train models
        self.model_normal = xgb.XGBRegressor(**xgb_params)
        self.model_spike = xgb.XGBRegressor(**xgb_params)
        
        self.model_normal.fit(Xn_tr, yn_tr)
        self.model_spike.fit(Xs_tr, ys_tr)
        
        # Evaluate
        norm_preds = self.model_normal.predict(Xn_te)
        spike_preds = self.model_spike.predict(Xs_te)
        
        norm_rmse = np.sqrt(mean_squared_error(yn_te, norm_preds))
        spike_rmse = np.sqrt(mean_squared_error(ys_te, spike_preds))
        
        norm_r2 = r2_score(yn_te, norm_preds)
        spike_r2 = r2_score(ys_te, spike_preds)
        
        print(f"Normal regime - RMSE: {norm_rmse:.2f}, R²: {norm_r2:.4f}")
        print(f"Spike regime - RMSE: {spike_rmse:.2f}, R²: {spike_r2:.4f}")
        
        # Save models
        self.save_models()
        
        return {
            'normal_rmse': norm_rmse,
            'normal_r2': norm_r2,
            'spike_rmse': spike_rmse,
            'spike_r2': spike_r2
        }
    
    def save_models(self):
        """Save trained models to disk"""
        joblib.dump(self.model_normal, os.path.join(self.regime_dir, 'model_normal.pkl'))
        joblib.dump(self.model_spike, os.path.join(self.regime_dir, 'model_spike.pkl'))
        
        # Save threshold value
        with open(os.path.join(self.regime_dir, 'threshold.txt'), 'w') as f:
            f.write(str(self.threshold))
            
        print(f"Models saved to {self.regime_dir}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.model_normal = joblib.load(os.path.join(self.regime_dir, 'model_normal.pkl'))
            self.model_spike = joblib.load(os.path.join(self.regime_dir, 'model_spike.pkl'))
            
            # Load threshold
            with open(os.path.join(self.regime_dir, 'threshold.txt'), 'r') as f:
                self.threshold = float(f.read().strip())
                
            print(f"Models loaded from {self.regime_dir}")
            print(f"Threshold for spike regime: {self.threshold:.2f} €/MWh")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, forecast_file, output_date=None):
        """
        Make predictions using the regime-based model.
        
        Args:
            forecast_file: CSV file with forecast data
            output_date: Date string for filtering and naming output file (optional)
            
        Returns:
            DataFrame with predictions
        """
        if self.model_normal is None or self.model_spike is None:
            if not self.load_models():
                raise ValueError("Models not trained or loaded")
        
        # Load data
        df = pd.read_csv(os.path.join(self.data_dir, forecast_file), parse_dates=['date'])
        df = self._engineer_features(df)
        
        # Filter by date if provided
        if output_date:
            df = df.loc[df['date'].dt.strftime('%Y-%m-%d') == output_date].copy()
        
        # Prepare feature matrix
        features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        X = df[features]
        
        # Determine regime for each sample
        is_spike = X['EP_lag_1'] > self.threshold
        
        # Predict using appropriate model for each regime
        df['Predicted'] = np.nan
        df.loc[~is_spike, 'Predicted'] = self.model_normal.predict(X.loc[~is_spike])
        df.loc[is_spike, 'Predicted'] = self.model_spike.predict(X.loc[is_spike])
        
        # Calculate prediction accuracy
        df['True'] = df['Electricity_price_MWh']
        df['Pct_of_True'] = df['Predicted'] / df['True'] * 100
        
        # Save results if output_date is provided
        if output_date:
            output_file = os.path.join(self.data_dir, f'{self.mapcode}_{output_date}_forecast.csv')
            df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']].to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]


# Example usage
if __name__ == "__main__":
    # Training example
    # model = XGBoostRegimeModel(mapcode="DK1")
    # model.train(training_file="DK1_full_data_2018_2024.csv")
    
    # Prediction example
    model = XGBoostRegimeModel(mapcode="DK1")
    results = model.predict(
        forecast_file="DK1_full_data_2025.csv",
        output_date="2025-03-01"
    )
    print(results)

"""
NOTES:
-----
This consolidated XGBoostRegimeModel is the preferred implementation for electricity price forecasting.
It uses a regime-switching approach that handles price spikes more effectively.

Key components:
1. The model maintains two separate XGBoost models:
   - One for normal pricing conditions
   - One for price spike conditions (defined as prices above the 90th percentile)

2. The regime is determined by the 1-hour lagged price value (EP_lag_1), 
   allowing the model to anticipate price spikes.

3. Feature engineering includes:
   - Cyclical hour encoding (sin/cos)
   - Rolling price and load statistics
   - Gas-load interaction term
   
4. The model automatically saves both sub-models to the regime_models directory.

The original implementation was based on XGBoost_regime_model.py and XGB_forecast_regime.py,
but has been consolidated into this single class for easier maintenance and use.
"""
