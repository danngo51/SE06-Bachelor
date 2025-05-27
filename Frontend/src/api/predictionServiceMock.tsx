import { PredictionDataResponse, HourlyPredictionData } from '../data/predictionTypes';

/**
 * Mock function to generate prediction data for testing
 * @param requestData Object containing country_codes array, date string, and weights object
 * @returns Promise with mock prediction data for generating graphs
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
  // Provide default values if parameters are invalid
  const safeCountryCodes = requestData.country_codes && requestData.country_codes.length > 0 
    ? requestData.country_codes 
    : ['XX'];
  const safeDate = requestData.date || new Date().toISOString().split('T')[0];
  const safeWeights = requestData.weights || { gru: 0.0, informer: 0.0, xgboost: 1.0 };
  
  // For development, use mock data instead of actual API calls
  console.log(`Mock data generated for ${safeCountryCodes.join(', ')} on ${safeDate} with weights:`, safeWeights);
  
  // Generate mock hourly data
  const generateHourlyDataForCountry = (code: string) => {
    const hourlyData: { [hour: string]: HourlyPredictionData } = {};
    
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
        informer: baseValue * hourlyPattern + Math.random() * 5,
        gru: baseValue * hourlyPattern + Math.random() * 7,
        xgboost: baseValue * hourlyPattern + Math.random() * 6,
        model: baseValue * hourlyPattern + Math.random() * 10,
        actual_price: baseValue * hourlyPattern * (1 + (Math.random() * 0.2 - 0.1)) + dateFactor
      };
    }
    
    return hourlyData;
  };
  
  // Create an array of country data objects
  const countries = safeCountryCodes.map(countryCode => ({
    countryCode: countryCode,
    hourlyData: generateHourlyDataForCountry(countryCode)
  }));
  
  // If one of the countryCodes is "ALL", add some additional mock countries
  if (safeCountryCodes.includes("ALL")) {
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

