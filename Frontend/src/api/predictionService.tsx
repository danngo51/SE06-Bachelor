import axios from 'axios';
import { toast } from 'react-toastify';
import { PredictionDataResponse } from '../data/predictionTypes';
import { API_ENDPOINTS, API_PARAMS } from '../data/constants';

// Base URL from environment or default
const API_BASE_URL =  'http://127.0.0.1:8000';

/**
 * Fetches prediction data for one or more countries and a specific date from the backend API
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
    
    // Call the backend API with the new JSON format
    const response = await axios.post(
      `${API_BASE_URL}${API_ENDPOINTS.PREDICTIONS}`, 
      {
        country_codes: [safeCountryCode], // Wrap the country code in an array
        date: safeDate
      }
    );
    
    // Data validation
    const data = response.data;
    if (!data.predictionDate || !data.countries || !Array.isArray(data.countries)) {
      throw new Error('Invalid data format returned from API');
    }
    
    // Validate that each country has hourly data
    for (const country of data.countries) {
      if (!country.countryCode || !country.hourlyData) {
        throw new Error('Invalid country data format returned from API');
      }
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching prediction data:', error);
    toast.error(`Error fetching prediction data for ${countryCode}`);
    throw error;
  }
};

