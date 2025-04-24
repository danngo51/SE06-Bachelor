// Create a utility to determine which service to use
// src/utils/apiSelector.tsx
import { fetchPredictionData as mockFetchPredictionData } from '../api/predictionServiceMock';
import { fetchPredictionData as realFetchPredictionData } from '../api/predictionService';
import { PredictionDataResponse } from '../api/predictionService'; // Use the same interface

// Check environment variable or local storage setting
const useMockApi = true
// Export the appropriate implementation
export const fetchPredictionData = useMockApi ? mockFetchPredictionData : realFetchPredictionData;
export type { PredictionDataResponse };