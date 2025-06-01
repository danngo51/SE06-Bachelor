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

        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.xgboost_dir = self.data_dir / "xgboost"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.xgboost_dir, exist_ok=True)

    def _engineer_features(self, df):
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'Electricity_price_MWh' not in df.columns:
            raise ValueError("Required column 'Electricity_price_MWh' not found in data")

        df['price_roll_3h'] = df['Electricity_price_MWh'].shift(1).rolling(3, min_periods=1).mean()
        df['EP_lag_1'] = df['Electricity_price_MWh'].shift(1)

        load_columns = ['DAHTL_TotalLoadValue', 'TotalLoadValue', 'Load_Value', 'Total_Load', 'DALoadValue', 'SystemLoad']
        load_col = next((col for col in load_columns if col in df.columns), None)
        df['load_roll_6h'] = df[load_col].shift(1).rolling(6, min_periods=1).mean() if load_col else 0

        gas_columns = ['Natural_Gas_price_EUR', 'Gas_Price']
        gas_col = next((col for col in gas_columns if col in df.columns), None)
        df['gas_x_load'] = df[gas_col] * df[load_col] if gas_col and load_col else 0

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

        expected_ftcs = [f"ftc_{conn}" for conn in country_connections.get(self.mapcode, [])]
        for ftc in expected_ftcs:
            if ftc not in df.columns:
                df[ftc] = 0

        required_fossil_cols = ['Fossil_Hard_coal_Capacity', 'Fossil_Hard_coal_Output', 'Fossil_Hard_coal_Utilization']
        for col in required_fossil_cols:
            if col not in df.columns:
                df[col] = 0

        return df.dropna()

    def train(self, training_file, xgb_params=None):
        xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'random_state': 42
        }

        df = pd.read_csv(os.path.join(self.data_dir, training_file), parse_dates=['date'])
        self.threshold = df['Electricity_price_MWh'].quantile(self.threshold_percentile)

        mask_spike = df['EP_lag_1'] > self.threshold
        mask_normal = ~mask_spike

        df_normal = df[mask_normal].copy()
        df_spike = df[mask_spike].copy()

        features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        X_norm, y_norm = df_normal[features], df_normal['Electricity_price_MWh']
        X_spk, y_spk = df_spike[features], df_spike['Electricity_price_MWh']

        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_norm, y_norm, test_size=0.2, shuffle=False)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_spk, y_spk, test_size=0.2, shuffle=False)

        self.model_normal = xgb.XGBRegressor(**xgb_params)
        self.model_spike = xgb.XGBRegressor(**xgb_params)

        self.model_normal.fit(Xn_tr, yn_tr)
        self.model_spike.fit(Xs_tr, ys_tr)

        norm_preds = self.model_normal.predict(Xn_te)
        spike_preds = self.model_spike.predict(Xs_te)

        metrics = {
            'normal_rmse': np.sqrt(mean_squared_error(yn_te, norm_preds)),
            'normal_mae': mean_absolute_error(yn_te, norm_preds),
            'normal_r2': r2_score(yn_te, norm_preds),
            'spike_rmse': np.sqrt(mean_squared_error(ys_te, spike_preds)),
            'spike_mae': mean_absolute_error(ys_te, spike_preds),
            'spike_r2': r2_score(ys_te, spike_preds)
        }

        metrics_df = pd.DataFrame({
            'Regime': ['Normal', 'Spike'],
            'RMSE': [metrics['normal_rmse'], metrics['spike_rmse']],
            'MAE': [metrics['normal_mae'], metrics['spike_mae']],
            'R²': [metrics['normal_r2'], metrics['spike_r2']]
        })
        metrics_df.to_csv(os.path.join(self.xgboost_dir, 'metrics.csv'), index=False)

        self.save_models()
        return metrics

    def save_models(self):
        joblib.dump(self.model_normal, os.path.join(self.xgboost_dir, 'model_normal.pkl'))
        joblib.dump(self.model_spike, os.path.join(self.xgboost_dir, 'model_spike.pkl'))
        with open(os.path.join(self.xgboost_dir, 'threshold.txt'), 'w') as f:
            f.write(str(self.threshold))

    def load_models(self):
        self.model_normal = joblib.load(os.path.join(self.xgboost_dir, 'model_normal.pkl'))
        self.model_spike = joblib.load(os.path.join(self.xgboost_dir, 'model_spike.pkl'))
        with open(os.path.join(self.xgboost_dir, 'threshold.txt'), 'r') as f:
            self.threshold = float(f.read().strip())

    def predict(self, forecast_file, output_date=None):
        if self.model_normal is None or self.model_spike is None:
            self.load_models()

        df = pd.read_csv(os.path.join(self.data_dir, forecast_file), parse_dates=['date'])
        if output_date:
            df = df[df['date'].dt.strftime('%Y-%m-%d') == output_date]

        for feature in self.model_normal.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0

        X = df[self.model_normal.feature_names_in_]
        is_spike = X['EP_lag_1'] > self.threshold

        df['Predicted'] = np.nan
        df.loc[~is_spike, 'Predicted'] = self.model_normal.predict(X.loc[~is_spike])
        df.loc[is_spike, 'Predicted'] = self.model_spike.predict(X.loc[is_spike])

        return df[['date', 'hour', 'Electricity_price_MWh', 'Predicted']]