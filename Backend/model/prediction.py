from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from datetime import date as date_type

class PredictionRequest(BaseModel):
    """Request model for prediction endpoints"""
    country_codes: List[str] = Field(
        ..., 
        description="List of ISO country codes to get predictions for", 
        example=["DK", "DE", "SE"]
    )
    date: Optional[str] = Field(
        None, 
        description="Date for the prediction in YYYY-MM-DD format", 
        example="2025-04-25"
    )
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional weights for each model type (xgboost, gru, informer). If not provided, default weights will be used.",
        example={"xgboost": 0.4, "gru": 0.3, "informer": 0.3}
    )

class HourlyData(BaseModel):
    """Hourly data for a country as expected by the frontend"""
    informer: float = Field(..., description="Prediction from the Informer model")
    gru: float = Field(..., description="Prediction from the GRU model")
    xgboost: float = Field(..., description="Prediction from the XGBoost model")
    model: float = Field(..., description="Combined model prediction")
    actual_price: Optional[float] = Field(None, description="Actual price (if available)")

class CountryData(BaseModel):
    """Country data with hourly predictions as expected by the frontend"""
    countryCode: str = Field(..., description="ISO country code")
    hourlyData: Dict[str, HourlyData] = Field(
        ..., 
        description="Hourly data indexed by hour (0-23)"
    )

class FrontendPredictionResponse(BaseModel):
    """Response model formatted for the frontend"""
    predictionDate: str = Field(..., description="Date of prediction in YYYY-MM-DD format")
    countries: List[CountryData] = Field(..., description="List of country data")
    
    class Config:
        schema_extra = {
            "example": {
                "predictionDate": "2025-04-24",
                "countries": [
                    {
                        "countryCode": "DE",                        "hourlyData": {                            "0": {"informer": 45.2, "gru": 44.8, "xgboost": 45.3, "model": 45.0, "actual_price": 47.5},
                            "1": {"informer": 42.8, "gru": 43.1, "xgboost": 43.0, "model": 42.9, "actual_price": 44.1},
                            # ... other hours
                        }
                    },
                    {
                        "countryCode": "DK1",                        "hourlyData": {                            "0": {"informer": 45.2, "gru": 45.5, "xgboost": 45.4, "model": 45.3, "actual_price": 45.5},
                            "1": {"informer": 45.8, "gru": 46.1, "xgboost": 45.7, "model": 45.9, "actual_price": 46.1},
                            # ... other hours
                        }
                    }
                ]
            }
        }

# Keep the existing models for internal use
class HourlyPredictionData(BaseModel):
    """Hourly prediction data with different model results (internal use)"""
    actual_price: Optional[float] = Field(None, description="Actual price (if available)")
    informer: float = Field(..., description="Prediction from the Informer model")
    gru: float = Field(..., description="Prediction from the GRU model")
    xgboost: float = Field(..., description="Prediction from the XGBoost model")
    model: float = Field(..., description="Combined model prediction")

class CountryPredictionData(BaseModel):
    """Prediction data for a single country (internal use)"""
    hourly_data: Dict[str, HourlyPredictionData] = Field(
        ..., 
        description="Hourly prediction data indexed by hour (0-23)"
    )
    country_code: str = Field(..., description="ISO country code")
    prediction_date: str = Field(..., description="Date of prediction in YYYY-MM-DD format")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoints (internal use)"""
    predictions: List[CountryPredictionData] = Field(
        ..., 
        description="List of prediction data for each requested country"
    )
    
    # Convert internal model to frontend format
    def to_frontend_format(self) -> FrontendPredictionResponse:
        """Convert the internal prediction response to the frontend format"""
        if not self.predictions:
            return FrontendPredictionResponse(predictionDate="", countries=[])
        
        prediction_date = self.predictions[0].prediction_date
        countries = []
        
        for country_data in self.predictions:
            hourly_data = {}
            for hour, data in country_data.hourly_data.items():
                # Extract model-specific predictions
                hourly_data[hour] = HourlyData(
                    informer=data.informer,
                    gru=data.gru,
                    xgboost=data.xgboost,
                    model=data.model,
                    actual_price=data.actual_price
                )
            
            countries.append(CountryData(
                countryCode=country_data.country_code,
                hourlyData=hourly_data
            ))
        
        return FrontendPredictionResponse(
            predictionDate=prediction_date,
            countries=countries
        )

class HybridModelOutput(BaseModel):
    """Model representing the output from the hybrid price prediction model"""
    informer_prediction: List[float] = Field(..., description="Predictions from the Informer model")
    gru_prediction: List[float] = Field(..., description="Predictions from the GRU model")
    xgboost_prediction: List[float] = Field(..., description="Predictions from the XGBoost model")
    model_prediction: List[float] = Field(..., description="Combined model predictions")