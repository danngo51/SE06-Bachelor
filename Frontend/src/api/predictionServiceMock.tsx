import { PredictionDataResponse } from '../data/predictionTypes';
import { PREDICTION_DATA_SERIES } from '../data/constants';

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
  
  // Generate mock hourly data
  const generateHourlyDataForCountry = (code: string) => {
    const hourlyData: { [hour: string]: Record<string, number> } = {};
    
    // Safely calculate country factor
    let countryFactor = 200; // Default value
    try {
      if (code.length >= 2) {
        countryFactor = code.charCodeAt(0) + code.charCodeAt(1);
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
    
    for (let hour = 0; hour < 24; hour++) {
      // Create variation in the data
      const baseValue = 50 + Math.sin(hour / 3) * 20 + (countryFactor % 10);
      const hourlyPattern = hour < 7 || hour > 19 ? 0.7 : 1.3; // Lower at night, higher during day
      
      hourlyData[hour.toString()] = {
        [PREDICTION_DATA_SERIES.PREDICTION_MODEL]: baseValue * hourlyPattern + (Math.random() * 10),
        [PREDICTION_DATA_SERIES.ACTUAL_PRICE]: baseValue * hourlyPattern * (1 + (Math.random() * 0.2 - 0.1)) + dateFactor
      };
    }
    
    return hourlyData;
  };
  
  // Create an array of country data objects
  const countries = [
    {
      countryCode: safeCountryCode,
      hourlyData: generateHourlyDataForCountry(safeCountryCode)
    }
  ];
  
  // If the countryCode is "ALL", add some additional mock countries
  if (safeCountryCode === "ALL") {
    countries.push(
      {
        countryCode: "DE",
        hourlyData: generateHourlyDataForCountry("DE")
      },
      {
        countryCode: "DK1",
        hourlyData: generateHourlyDataForCountry("DK1")
      },
      {
        countryCode: "SE",
        hourlyData: generateHourlyDataForCountry("SE")
      }
    );
  }
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    predictionDate: safeDate,
    countries: countries
  };
};

