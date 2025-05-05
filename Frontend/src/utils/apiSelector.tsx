// Create a utility to determine which service to use
import { fetchPredictionData as mockFetchPredictionData } from '../api/predictionServiceMock';
import { fetchPredictionData as realFetchPredictionData } from '../api/predictionService';
import { PredictionDataResponse } from '../data/predictionTypes';

// Check environment variable or local storage setting
const useMockApi = false;

// Export the appropriate implementation
export const fetchPredictionData = useMockApi ? mockFetchPredictionData : realFetchPredictionData;
export type { PredictionDataResponse };