# XGBoost Models Package
"""
This module contains XGBoost regime-based models for electricity price forecasting.
The regime-based approach splits the electricity price data into normal and spike regimes,
training separate models for each to improve prediction accuracy.
"""

from .xgboost_pipeline import XGBoostPipeline

__all__ = ['XGBoostPipeline']