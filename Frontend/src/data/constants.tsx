/**
 * Constants for prediction data series names
 * Centralizing these names makes it easier to change them in one place
 */

// Data series names for prediction models
export const PREDICTION_DATA_SERIES = {
  PREDICTION_MODEL: 'Model',
  ACTUAL_PRICE: 'actual'
};

// Other constants related to the prediction feature
export const API_ENDPOINTS = {
  PREDICTIONS: '/api/predictions',
  AVAILABLE_DATES: '/api/predictions/available-dates'
};

// Parameter names for API requests
export const API_PARAMS = {
  COUNTRY: 'country',
  DATE: 'date'
};