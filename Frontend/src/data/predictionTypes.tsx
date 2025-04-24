/**
 * Type definitions for prediction data
 * These types define the structure of data returned from the prediction API
 */

// Type definition for prediction data response with hourly values
export interface HourlyPredictionData {
  [key: string]: number; // Allow dynamic keys for model names
}

export interface PredictionDataResponse {
  // We can remove the timestamp array since it can be constructed from predictionDate and hour keys
  hourlyData: {
    [hour: string]: HourlyPredictionData;
  };
  countryCode: string;
  predictionDate: string;
}