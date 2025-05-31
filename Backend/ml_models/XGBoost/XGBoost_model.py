import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class XGBoostRegimeModel:
    def __init__(self, mapcode="DK1", threshold_percentile=0.90):
        self.mapcode = mapcode
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.model_normal = None
        self.model_spike = None
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.xgboost_dir = self.data_dir / "xgboost"
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.xgboost_dir, exist_ok=True)
    
    def _engineer_features(self, df):
        # Create a copy to avoid modifying the original data
        df = df.copy()
        
        # Required time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Price features - required for regime split
        if 'Electricity_price_MWh' not in df.columns:
            raise ValueError("Required column 'Electricity_price_MWh' not found in data")
        
        df['price_roll_3h'] = df['Electricity_price_MWh'].shift(1).rolling(3, min_periods=1).mean()
        df['EP_lag_1'] = df['Electricity_price_MWh'].shift(1)
        
        # Load features - try to find available load column
        load_columns = [
            'DAHTL_TotalLoadValue', 
            'TotalLoadValue', 
            'Load_Value', 
            'Total_Load',
            'DALoadValue',
            'SystemLoad'
        ]
        load_col = next((col for col in load_columns if col in df.columns), None)
        if load_col:
            df['load_roll_6h'] = df[load_col].shift(1).rolling(6, min_periods=1).mean()
        else:
            print(f"Warning: No load column found in {df.columns.tolist()}. Load features will not be generated.")
            df['load_roll_6h'] = 0
            
        # Gas-load interaction if gas price available
        gas_columns = ['Natural_Gas_price_EUR', 'Gas_Price']
        gas_col = next((col for col in gas_columns if col in df.columns), None)
        
        if gas_col and load_col:
            df['gas_x_load'] = df[gas_col] * df[load_col]
        else:
            df['gas_x_load'] = 0
            if gas_col:
                print("Warning: Load column not found for gas-load interaction")
            elif load_col:
                print("Warning: Gas price column not found for gas-load interaction")
                
        # Handle Flow Transfer Capacity (FTC) columns dynamically
        ftc_cols = [col for col in df.columns if col.startswith('ftc_')]
        
        # Define known interconnections for each country/bidding zone
        country_connections = {
            "DK1": ["DE_LU", "NO2", "SE3", "NL"],
            "DK2": ["DE_LU", "SE4"],
            "SE1": ["SE2", "FI", "NO4"],
            "SE2": ["SE3", "NO4", "NO3"],
            "SE3": ["SE4", "NO1", "FI", "DK1"],
            "SE4": ["DK2", "DE_LU", "PL"],
            "NO1": ["NO2", "NO3", "NO5", "SE3"],
            "NO2": ["NO5", "DK1", "DE_LU", "NL"],
            "NO3": ["NO4", "NO5", "SE2"],
            "NO4": ["SE1", "SE2", "NO3", "FI"],
            "NO5": ["NO1", "NO2", "NO3"],
            "FI": ["SE1", "SE3", "NO4", "EE"],
            "NL": ["DE_LU", "NO2", "DK1", "BE"],
            "BE": ["FR", "NL", "DE_LU"],
            "FR": ["BE", "DE_LU", "ES", "IT-NORTH"],
            "DE_LU": ["DK1", "DK2", "SE4", "NO2", "NL", "BE", "FR", "CH", "AT"],
        }

        # Get expected FTC columns for this country
        if self.mapcode in country_connections:
            expected_ftcs = [f"ftc_{conn}" for conn in country_connections[self.mapcode]]
            
            # Add missing FTC columns with zeros
            for ftc in expected_ftcs:
                if ftc not in df.columns:
                    print(f"Adding missing FTC column {ftc} with zeros for {self.mapcode}")
                    df[ftc] = 0
        else:
            print(f"Warning: No predefined FTC connections found for {self.mapcode}")
            
        # Handle fossil fuel columns
        required_fossil_cols = [
            'Fossil_Hard_coal_Capacity',
            'Fossil_Hard_coal_Output',
            'Fossil_Hard_coal_Utilization'
        ]
        
        for col in required_fossil_cols:
            if col not in df.columns:
                print(f"Warning: Adding missing column {col} with zeros")
                df[col] = 0
                
        return df.dropna()
    
    def train(self, training_file, xgb_params=None):
        # Default XGBoost parameters if none provided
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'random_state': 42
            }
        
        # Load and prepare training data
        df = pd.read_csv(os.path.join(self.data_dir, training_file), parse_dates=['date'])
        # df = self._engineer_features(df)
        
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
        
        norm_mae = mean_absolute_error(yn_te, norm_preds)
        spike_mae = mean_absolute_error(ys_te, spike_preds)
        
        non_zero_mask = yn_te != 0
        norm_mape = np.mean(np.abs((yn_te[non_zero_mask] - norm_preds[non_zero_mask]) / yn_te[non_zero_mask])) * 100
        spike_mape = np.mean(np.abs((ys_te - spike_preds) / ys_te)) * 100
        
        norm_r2 = r2_score(yn_te, norm_preds)
        spike_r2 = r2_score(ys_te, spike_preds)
        
        print(f"Normal regime - RMSE: {norm_rmse:.2f}, MAE: {norm_mae:.2f}, MAPE: {norm_mape:.2f}%, R²: {norm_r2:.4f}")
        print(f"Spike regime - RMSE: {spike_rmse:.2f}, MAE: {spike_mae:.2f}, MAPE: {spike_mape:.2f}%, R²: {spike_r2:.4f}")
        
        # Save metrics to CSV
        metrics_file = os.path.join(self.xgboost_dir, 'metrics.csv')
        metrics_data = {
            'Regime': ['Normal', 'Spike'],
            'RMSE': [norm_rmse, spike_rmse],
            'MAE': [norm_mae, spike_mae],
            'MAPE (%)': [norm_mape, spike_mape],
            'R²': [norm_r2, spike_r2]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
        
        # Save models
        self.save_models()
        
        return {
            'normal_rmse': norm_rmse,
            'normal_mae': norm_mae,
            'normal_mape': norm_mape,
            'normal_r2': norm_r2,
            'spike_rmse': spike_rmse,
            'spike_mae': spike_mae,
            'spike_mape': spike_mape,
            'spike_r2': spike_r2
        }
    
    def save_models(self):
        """Save trained models and feature information to disk"""
        joblib.dump(self.model_normal, os.path.join(self.xgboost_dir, 'model_normal.pkl'))
        joblib.dump(self.model_spike, os.path.join(self.xgboost_dir, 'model_spike.pkl'))
        
        # Save threshold value
        with open(os.path.join(self.xgboost_dir, 'threshold.txt'), 'w') as f:
            f.write(str(self.threshold))
            
        print(f"Models saved to {self.xgboost_dir}")

        # Save list of expected features
        features = self.model_normal.feature_names_in_.tolist()
        with open(os.path.join(self.xgboost_dir, 'features.txt'), 'w') as f:
            f.write('\n'.join(features))
    
    def load_models(self):
        """Load trained models and feature information from disk"""
        try:
            self.model_normal = joblib.load(os.path.join(self.xgboost_dir, 'model_normal.pkl'))
            self.model_spike = joblib.load(os.path.join(self.xgboost_dir, 'model_spike.pkl'))
            
            # Load threshold
            with open(os.path.join(self.xgboost_dir, 'threshold.txt'), 'r') as f:
                self.threshold = float(f.read().strip())
                
            # Load expected features
            try:
                with open(os.path.join(self.xgboost_dir, 'features.txt'), 'r') as f:
                    self.expected_features = f.read().splitlines()
            except:
                # If features.txt doesn't exist, use feature names from model
                self.expected_features = self.model_normal.feature_names_in_.tolist()
                
            print(f"Models loaded from {self.xgboost_dir}")
            print(f"Threshold for spike regime: {self.threshold:.2f} €/MWh")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, forecast_file, output_date=None):
        if self.model_normal is None or self.model_spike is None:
            if not self.load_models():
                raise ValueError("Models not trained or loaded")
        
        # Load data
        df = pd.read_csv(os.path.join(self.data_dir, forecast_file), parse_dates=['date'])
        # df = self._engineer_features(df)
        
        # Filter by date if provided
        if output_date:
            df = df.loc[df['date'].dt.strftime('%Y-%m-%d') == output_date].copy()
        
        # Ensure all expected features are present
        for feature in self.expected_features:
            if feature not in df.columns:
                print(f"Adding missing feature {feature} with zeros")
                df[feature] = 0
        
        # Use only the expected features in the correct order
        X = df[self.expected_features]
        
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

