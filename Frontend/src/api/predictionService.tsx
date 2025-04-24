import axios from 'axios';
import { toast } from 'react-toastify';

// Base URL from environment or default
const API_BASE_URL = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000';

// Type definition for prediction data response with hourly values
export interface HourlyPredictionData {
  'Prediction Informer': number;
  'actual': number;
  'tbd': number;
}

export interface PredictionDataResponse {
  timestamp: string[];
  hourlyData: {
    [hour: string]: HourlyPredictionData;
  };
  countryCode: string;
  predictionDate: string;
}

/**
 * Fetches prediction data for a specific country and date from the backend API
 * @param countryCode The ISO country code (e.g., 'DK', 'DE')
 * @param date The date for prediction in YYYY-MM-DD format
 * @returns Promise with prediction data for generating graphs
 */
export const fetchPredictionData = async (
  countryCode: string, 
  date: string
): Promise<PredictionDataResponse> => {
  try {
    // Make sure we have valid parameters
    const safeCountryCode = countryCode || 'XX';
    const safeDate = date || new Date().toISOString().split('T')[0];
    
    console.log(`Fetching prediction data for ${safeCountryCode} on ${safeDate}`);
    
    // Call the backend API
    const response = await axios.get(
      `${API_BASE_URL}/api/predictions`, 
      { 
        params: { 
          country: safeCountryCode,
          date: safeDate 
        } 
      }
    );
    
    // Data validation
    const data = response.data;
    if (!data.timestamp || !data.hourlyData) {
      throw new Error('Invalid data format returned from API');
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching prediction data:', error);
    toast.error(`Error fetching prediction data for ${countryCode}`);
    throw error;
  }
};

/**
 * Fetches available prediction dates for a country
 * This could be useful if you want to limit date selection to available dates
 */
export const fetchAvailablePredictionDates = async (
  countryCode: string
): Promise<string[]> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/api/predictions/available-dates`,
      {
        params: {
          country: countryCode
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error fetching available prediction dates:', error);
    return []; // Return empty array if dates can't be fetched
  }
};