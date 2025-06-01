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
        self.feature_dir = self.data_dir / "feature"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.xgboost_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    def train(self, training_file, xgb_params=None):
        xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'random_state': 42
        }

        # Load data
        df = pd.read_csv(os.path.join(self.data_dir, training_file), parse_dates=['date'])
        self.threshold = df['Electricity_price_MWh'].quantile(self.threshold_percentile)

        # Split data into normal and spike regimes
        mask_spike = df['Electricity_price_MWh'] > self.threshold
        mask_normal = ~mask_spike

        df_normal = df[mask_normal].copy()
        df_spike = df[mask_spike].copy()

        # Define features and targets
        features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        X_norm, y_norm = df_normal[features], df_normal['Electricity_price_MWh']
        X_spk, y_spk = df_spike[features], df_spike['Electricity_price_MWh']

        # Split into training and testing sets
        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_norm, y_norm, test_size=0.2, shuffle=False)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_spk, y_spk, test_size=0.2, shuffle=False)

        # Train models
        self.model_normal = xgb.XGBRegressor(**xgb_params)
        self.model_spike = xgb.XGBRegressor(**xgb_params)

        self.model_normal.fit(Xn_tr, yn_tr)
        self.model_spike.fit(Xs_tr, ys_tr)

        # Evaluate models
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

        # Save models and top features
        self.save_models()
        self.save_top_features(features)
        return metrics

    def save_models(self):
        joblib.dump(self.model_normal, os.path.join(self.xgboost_dir, 'model_normal.pkl'))
        joblib.dump(self.model_spike, os.path.join(self.xgboost_dir, 'model_spike.pkl'))
        with open(os.path.join(self.xgboost_dir, 'threshold.txt'), 'w') as f:
            f.write(str(self.threshold))

    def save_top_features(self, features):
        # Combine feature importance from both models
        importance_normal = pd.Series(self.model_normal.feature_importances_, index=features)
        importance_spike = pd.Series(self.model_spike.feature_importances_, index=features)
        combined_importance = (importance_normal + importance_spike).sort_values(ascending=False)

        # Normalize importance to percentages
        total_importance = combined_importance.sum()
        combined_importance_percentage = (combined_importance / total_importance) * 100

        # Select top 10 features
        top_features = combined_importance_percentage.head(10).reset_index()
        top_features.columns = ['Feature', 'Importance (%)']

        # Save to CSV
        top_features.to_csv(os.path.join(self.feature_dir, 'features.csv'), index=False)
        print("Top 10 features saved to features.csv.")

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
        is_spike = X['Electricity_price_MWh'] > self.threshold

        df['Predicted'] = np.nan
        df.loc[~is_spike, 'Predicted'] = self.model_normal.predict(X.loc[~is_spike])
        df.loc[is_spike, 'Predicted'] = self.model_spike.predict(X.loc[is_spike])

        return df[['date', 'hour', 'Electricity_price_MWh', 'Predicted']]