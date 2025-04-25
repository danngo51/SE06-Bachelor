/**
 * Type definitions for prediction data
 * These types define the structure of data returned from the prediction API
 */

// Type definition for prediction data response with hourly values
export interface HourlyPredictionData {
  [key: string]: number; // Allow dynamic keys for model names
}

// Structure for a single country's prediction data
export interface CountryPredictionData {
  countryCode: string;
  hourlyData: {
    [hour: string]: HourlyPredictionData;
  };
}

export interface PredictionDataResponse {
  predictionDate: string;
  countries: CountryPredictionData[];
}