import axios from 'axios';
import { toast } from 'react-toastify';

// Base URL from environment or default
const API_BASE_URL = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000/';

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
 * Mock function to generate prediction data for testing
 * @param countryCode The ISO country code (e.g., 'DK', 'DE')
 * @param date The date for prediction in YYYY-MM-DD format
 * @returns Promise with mock prediction data for generating graphs
 */
export const fetchPredictionData = async (
  countryCode: string, 
  date: string
): Promise<PredictionDataResponse> => {
  // Provide default values if parameters are invalid
  const safeCountryCode = countryCode || 'XX';
  const safeDate = date || new Date().toISOString().split('T')[0];
  
  // For development, use mock data instead of actual API calls
  console.log(`Mock data generated for ${safeCountryCode} on ${safeDate}`);
  
  // Create a timestamp array for the 24 hours of the selected date
  const timestamps = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, '0');
    return `${safeDate}T${hour}:00:00`;
  });
  
  // Generate mock hourly data
  const hourlyData: { [hour: string]: HourlyPredictionData } = {};
  
  for (let hour = 0; hour < 24; hour++) {
    // Safely calculate country factor
    let countryFactor = 200; // Default value
    try {
      if (safeCountryCode.length >= 2) {
        countryFactor = safeCountryCode.charCodeAt(0) + safeCountryCode.charCodeAt(1);
      }
    } catch (error) {
      console.warn('Error calculating country factor, using default', error);
    }
    
    // Safely parse date parts
    let dateFactor = 10; // Default value
    try {
      const dateParts = safeDate.split('-');
      if (dateParts.length >= 3) {
        dateFactor = parseInt(dateParts[2]) * 0.5; // Day of month affects the pattern
      }
    } catch (error) {
      console.warn('Error calculating date factor, using default', error);
    }
    
    // Create variation in the data
    const baseValue = 50 + Math.sin(hour / 3) * 20 + (countryFactor % 10);
    const hourlyPattern = hour < 7 || hour > 19 ? 0.7 : 1.3; // Lower at night, higher during day
    
    hourlyData[hour.toString()] = {
      'Prediction Informer': baseValue * hourlyPattern + (Math.random() * 10),
      'actual': baseValue * hourlyPattern * (1 + (Math.random() * 0.2 - 0.1)) + dateFactor,
      'tbd': baseValue * 0.8 * hourlyPattern + (Math.sin(hour) * 5)
    };
  }
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    timestamp: timestamps,
    hourlyData: hourlyData,
    countryCode: safeCountryCode,
    predictionDate: safeDate
  };
};

/**
 * Mock function to fetch available prediction dates
 * @param countryCode The ISO country code
 * @returns Array of dates for which predictions are available
 */
export const fetchAvailablePredictionDates = async (
  countryCode: string
): Promise<string[]> => {
  // Generate 10 dates starting from today
  const today = new Date();
  const dates: string[] = [];
  
  for (let i = 0; i < 10; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
  }
  
  console.log(`Mock available dates generated for ${countryCode}:`, dates);
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300));
  
  return dates;
};