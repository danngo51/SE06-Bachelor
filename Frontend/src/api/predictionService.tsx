import axios from 'axios';
import { toast } from 'react-toastify';
import { PredictionDataResponse } from '../data/predictionTypes';
import { API_ENDPOINTS } from '../data/constants';

// Base URL from environment or default
const API_BASE_URL =  'http://127.0.0.1:8000';

/**
 * Fetches prediction data for one or more countries and a specific date from the backend API
 * @param requestData Object containing country_codes array, date string, and weights object
 * @returns Promise with prediction data for generating graphs
 */
export const fetchPredictionData = async (
  requestData: {
    country_codes: string[];
    date: string;
    weights: {
      gru: number;
      informer: number;
      xgboost: number;
    };
  }
): Promise<PredictionDataResponse> => {
  try {
    // Validate the request data
    const safeCountryCodes = requestData.country_codes && requestData.country_codes.length > 0 
      ? requestData.country_codes 
      : ['XX'];
    const safeDate = requestData.date || new Date().toISOString().split('T')[0];
    const safeWeights = requestData.weights || { gru: 0.0, informer: 0.0, xgboost: 1.0 };
    
    console.log(`Fetching prediction data for ${safeCountryCodes.join(', ')} on ${safeDate} with weights:`, safeWeights);
    
    // Call the backend API with the new JSON format
    const response = await axios.post(
      `${API_BASE_URL}${API_ENDPOINTS.PREDICTIONS}`, 
      {
        country_codes: safeCountryCodes,
        date: safeDate,
        weights: safeWeights
      }
    );
    
    // Data validation
    const data = response.data;
    if (!data.predictionDate || !data.countries || !Array.isArray(data.countries)) {
      throw new Error('Invalid data format returned from API');
    }
    
    // Validate that each country has hourly data with the expected model fields
    for (const country of data.countries) {
      if (!country.countryCode || !country.hourlyData) {
        throw new Error('Invalid country data format returned from API');
      }
      
      // Check at least one hour entry to validate the structure has all required model fields
      const hourKeys = Object.keys(country.hourlyData);
      if (hourKeys.length > 0) {
        const sampleHourData = country.hourlyData[hourKeys[0]];        if (!('informer' in sampleHourData && 
              'gru' in sampleHourData && 
              'xgboost' in sampleHourData && 
              'model' in sampleHourData && 
              'actual_price' in sampleHourData)) {
          console.warn('Missing expected model fields in hourly data', sampleHourData);
        }
      }
    }
    
    return data;  } catch (error) {
    console.error('Error fetching prediction data:', error);
    toast.error(`Error fetching prediction data`);
    throw error;
  }
};

