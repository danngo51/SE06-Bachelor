import React, { useState } from 'react';
import { 
  LineChart, Line, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import { PredictionDataResponse } from '../../utils/apiSelector';
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
 */
const PredictionGraphs: React.FC<PredictionGraphsProps> = ({ 
  data, 
  isVisible, 
  onClose, 
  countryName,
  predictionDate
}) => {
  const [activeGraph, setActiveGraph] = useState<'line' | 'bar' | 'area'>('line');

  if (!isVisible || !data) return null;
  
  // Process hourlyData for chart display
  const chartData = data.timestamp.map((time, index) => {
    const hour = index.toString();
    const hourData = data.hourlyData[hour] || { 'Prediction Informer': 0, 'actual': 0, 'tbd': 0 };
    
    return {
      time,
      'Prediction Informer': hourData['Prediction Informer'],
      'actual': hourData['actual'],
      'tbd': hourData['tbd'],
      hour: new Date(time).getHours()
    };
  });

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
        <button 
          onClick={() => setActiveGraph('area')}
          className={`${styles.graphTypeButton} ${activeGraph === 'area' ? styles.active : ''}`}
        >
          Area Chart
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
                formatter={(value) => [`${value.toFixed(2)}`, '']}
                labelFormatter={(hour) => `Hour: ${hour}:00`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="Prediction Informer" 
                stroke="#8884d8" 
                name="Prediction Model"
                strokeWidth={2}
                dot={{ strokeWidth: 2, r: 3 }}
                activeDot={{ r: 8 }} 
              />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#82ca9d" 
                name="Actual Price"
                strokeWidth={2}
                dot={{ strokeWidth: 2, r: 3 }}
                activeDot={{ r: 8 }} 
              />
              <Line 
                type="monotone" 
                dataKey="tbd" 
                stroke="#ffc658" 
                name="Additional Metric"
                strokeWidth={2}
                dot={{ strokeWidth: 2, r: 3 }}
                activeDot={{ r: 8 }} 
              />
            </LineChart>
          ) : activeGraph === 'bar' ? (
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="hour"
                tick={{ fontSize: 12 }}
                label={{ value: 'Hour of day', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Price', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value) => [`${value.toFixed(2)}`, '']}
                labelFormatter={(hour) => `Hour: ${hour}:00`}
              />
              <Legend />
              <Bar dataKey="Prediction Informer" fill="#8884d8" name="Prediction Model" />
              <Bar dataKey="actual" fill="#82ca9d" name="Actual Price" />
              <Bar dataKey="tbd" fill="#ffc658" name="Additional Metric" />
            </BarChart>
          ) : (
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="hour"
                tick={{ fontSize: 12 }}
                label={{ value: 'Hour of day', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Price', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value) => [`${value.toFixed(2)}`, '']}
                labelFormatter={(hour) => `Hour: ${hour}:00`}
              />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="Prediction Informer" 
                stroke="#8884d8" 
                fill="#8884d8"
                fillOpacity={0.3}
                name="Prediction Model" 
              />
              <Area 
                type="monotone" 
                dataKey="actual" 
                stroke="#82ca9d" 
                fill="#82ca9d"
                fillOpacity={0.3}
                name="Actual Price" 
              />
              <Area 
                type="monotone" 
                dataKey="tbd" 
                stroke="#ffc658" 
                fill="#ffc658"
                fillOpacity={0.3}
                name="Additional Metric" 
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
        <p className={styles.graphCaption}>
          Comparing the prediction model output with actual prices across 24 hours
        </p>
      </div>
    </div>
  );
};

export default PredictionGraphs;