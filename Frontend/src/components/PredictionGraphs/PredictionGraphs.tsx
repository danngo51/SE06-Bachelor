import React, { useState } from 'react';
import { 
  LineChart, Line, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer
} from 'recharts';
import { PredictionDataResponse, CountryPredictionData } from '../../data/predictionTypes';
import { PREDICTION_DATA_SERIES } from '../../data/constants';
import styles from './PredictionGraphs.module.css';

interface PredictionGraphsProps {
  data: PredictionDataResponse | null;
  isVisible: boolean;
  onClose: () => void;
  countryName: string;
  predictionDate: string;
}

/**
 * Component to display prediction data using various graph types
 * Now handles multiple countries by showing the first country in the array by default
 */
const PredictionGraphs: React.FC<PredictionGraphsProps> = ({ 
  data, 
  isVisible, 
  onClose, 
  countryName,
  predictionDate
}) => {
  const [activeGraph, setActiveGraph] = useState<'line' | 'bar'>('line');

  if (!isVisible || !data) return null;
  
  // Get the first country's data from the countries array
  const countryData = data.countries && data.countries.length > 0 
    ? data.countries[0] 
    : null;
    
  if (!countryData) return null;
  
  // Process hourlyData for chart display
  const chartData = Object.entries(countryData.hourlyData).map(([hour, hourData]) => {
    // Generate timestamp for each hour
    const hourNum = parseInt(hour);
    const formattedHour = hourNum.toString().padStart(2, '0');
    const timestamp = `${data.predictionDate}T${formattedHour}:00:00`;
    
    return {
      time: timestamp,
      [PREDICTION_DATA_SERIES.PREDICTION_MODEL]: hourData[PREDICTION_DATA_SERIES.PREDICTION_MODEL] || 0,
      [PREDICTION_DATA_SERIES.ACTUAL_PRICE]: hourData[PREDICTION_DATA_SERIES.ACTUAL_PRICE] || 0,
      hour: hourNum
    };
  }).sort((a, b) => a.hour - b.hour); // Sort by hour to ensure correct order

  return (
    <div className={styles.graphContainer}>
      <div className={styles.graphHeader}>
        <h3 className={styles.graphTitle}>
          Prediction for {countryName} on {predictionDate}
        </h3>
        <button 
          className={styles.closeButton} 
          onClick={onClose}
          aria-label="Close prediction graphs"
        >
          ×
        </button>
      </div>
      
      <div className={styles.graphControls}>
        <button 
          onClick={() => setActiveGraph('line')}
          className={`${styles.graphTypeButton} ${activeGraph === 'line' ? styles.active : ''}`}
        >
          Line Chart
        </button>
        <button 
          onClick={() => setActiveGraph('bar')}
          className={`${styles.graphTypeButton} ${activeGraph === 'bar' ? styles.active : ''}`}
        >
          Bar Chart
        </button>
      </div>
      
      <div className={styles.graphItem}>
        <ResponsiveContainer width="100%" height={300}>
          {activeGraph === 'line' ? (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="hour" 
                tick={{ fontSize: 12 }}
                label={{ value: 'Hour of day', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Price', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value) => (typeof value === 'number' ? [`${value.toFixed(2)}`, ''] : [value, ''])}
                labelFormatter={(hour) => `Hour: ${hour}:00`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey={PREDICTION_DATA_SERIES.PREDICTION_MODEL} 
                stroke="#8884d8" 
                name="Prediction Model"
                strokeWidth={2}
                dot={{ strokeWidth: 2, r: 3 }}
                activeDot={{ r: 8 }} 
              />
              <Line 
                type="monotone" 
                dataKey={PREDICTION_DATA_SERIES.ACTUAL_PRICE} 
                stroke="#82ca9d" 
                name="Actual Price"
                strokeWidth={2}
                dot={{ strokeWidth: 2, r: 3 }}
                activeDot={{ r: 8 }} 
              />
            </LineChart>
          ) : (
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="hour"
                tick={{ fontSize: 12 }}
                label={{ value: 'Hour of day', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Price', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value) => (typeof value === 'number' ? [`${value.toFixed(2)}`, ''] : [value, ''])}
                labelFormatter={(hour) => `Hour: ${hour}:00`}
              />
              <Legend />
              <Bar dataKey={PREDICTION_DATA_SERIES.PREDICTION_MODEL} fill="#8884d8" name="Prediction Model" />
              <Bar dataKey={PREDICTION_DATA_SERIES.ACTUAL_PRICE} fill="#82ca9d" name="Actual Price" />
            </BarChart>
          )}
        </ResponsiveContainer>
        <p className={styles.graphCaption}>
          Comparing prediction model output with actual prices across 24 hours
        </p>
      </div>
    </div>
  );
};

export default PredictionGraphs;