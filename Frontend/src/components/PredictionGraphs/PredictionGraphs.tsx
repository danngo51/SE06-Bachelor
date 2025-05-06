import React, { useState } from 'react';
import { 
  LineChart, Line, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer
} from 'recharts';
import { PredictionDataResponse, CountryPredictionData, HourlyPredictionData } from '../../data/predictionTypes';
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
 * Allows selecting which specific graphs to display
 */
const PredictionGraphs: React.FC<PredictionGraphsProps> = ({ 
  data, 
  isVisible, 
  onClose, 
  countryName,
  predictionDate
}) => {
  const [activeGraph, setActiveGraph] = useState<'line' | 'bar'>('line');
  
  // Track which models to display with individual toggles
  const [showInformer, setShowInformer] = useState<boolean>(true);
  const [showGru, setShowGru] = useState<boolean>(true);
  const [showCombined, setShowCombined] = useState<boolean>(true);
  const [showActual, setShowActual] = useState<boolean>(true);

  if (!isVisible || !data) return null;
  
  // Get the first country's data from the countries array
  const countryData = data.countries && data.countries.length > 0 
    ? data.countries[0] 
    : null;
    
  if (!countryData) return null;
  
  // Process hourlyData for chart display
  const chartData = Object.entries(countryData.hourlyData).map(([hour, hourData]: [string, HourlyPredictionData]) => {
    // Generate timestamp for each hour
    const hourNum = parseInt(hour);
    const formattedHour = hourNum.toString().padStart(2, '0');
    const timestamp = `${data.predictionDate}T${formattedHour}:00:00`;
    
    return {
      time: timestamp,
      [PREDICTION_DATA_SERIES.INFORMER_MODEL]: hourData[PREDICTION_DATA_SERIES.INFORMER_MODEL as keyof HourlyPredictionData] || 0,
      [PREDICTION_DATA_SERIES.GRU_MODEL]: hourData[PREDICTION_DATA_SERIES.GRU_MODEL as keyof HourlyPredictionData] || 0,
      [PREDICTION_DATA_SERIES.PREDICTION_MODEL]: hourData[PREDICTION_DATA_SERIES.PREDICTION_MODEL as keyof HourlyPredictionData] || 0,
      [PREDICTION_DATA_SERIES.ACTUAL_PRICE]: hourData[PREDICTION_DATA_SERIES.ACTUAL_PRICE as keyof HourlyPredictionData] || 0,
      hour: hourNum
    };
  }).sort((a, b) => a.hour - b.hour); // Sort by hour to ensure correct order

  // Helper function to toggle all graphs
  const toggleAllGraphs = (show: boolean) => {
    setShowInformer(show);
    setShowGru(show);
    setShowCombined(show);
    setShowActual(show);
  };

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

      <div className={styles.graphSelectionControls}>
        <div className={styles.selectionHeader}>
          <span>Select Models to Display:</span>
          <div>
            <button 
              onClick={() => toggleAllGraphs(true)}
              className={`${styles.smallButton} ${showInformer && showGru && showCombined && showActual ? styles.active : ''}`}
            >
              Show All
            </button>
            <button 
              onClick={() => toggleAllGraphs(false)}
              className={`${styles.smallButton} ${!showInformer && !showGru && !showCombined && !showActual ? styles.active : ''}`}
            >
              Hide All
            </button>
          </div>
        </div>

        <div className={styles.checkboxContainer}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showInformer}
              onChange={() => setShowInformer(!showInformer)}
            />
            <span className={styles.modelInformer}>Informer Model</span>
          </label>
          
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showGru}
              onChange={() => setShowGru(!showGru)}
            />
            <span className={styles.modelGru}>GRU Model</span>
          </label>
          
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showCombined}
              onChange={() => setShowCombined(!showCombined)}
            />
            <span className={styles.modelCombined}>Combined Model</span>
          </label>
          
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showActual}
              onChange={() => setShowActual(!showActual)}
            />
            <span className={styles.modelActual}>Actual Price</span>
          </label>
        </div>
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
              {showInformer && (
                <Line 
                  type="monotone" 
                  dataKey={PREDICTION_DATA_SERIES.INFORMER_MODEL} 
                  stroke="#FF8C00" 
                  name="Informer Model"
                  strokeWidth={1.5}
                  dot={{ strokeWidth: 1, r: 2 }}
                  activeDot={{ r: 6 }} 
                />
              )}
              {showGru && (
                <Line 
                  type="monotone" 
                  dataKey={PREDICTION_DATA_SERIES.GRU_MODEL} 
                  stroke="#8A2BE2" 
                  name="GRU Model"
                  strokeWidth={1.5}
                  dot={{ strokeWidth: 1, r: 2 }}
                  activeDot={{ r: 6 }} 
                />
              )}
              {showCombined && (
                <Line 
                  type="monotone" 
                  dataKey={PREDICTION_DATA_SERIES.PREDICTION_MODEL} 
                  stroke="#0066CC" 
                  name="Combined Model"
                  strokeWidth={2}
                  dot={{ strokeWidth: 2, r: 3 }}
                  activeDot={{ r: 8 }} 
                />
              )}
              {showActual && (
                <Line 
                  type="monotone" 
                  dataKey={PREDICTION_DATA_SERIES.ACTUAL_PRICE} 
                  stroke="#00CC66" 
                  name="Actual Price"
                  strokeWidth={2}
                  dot={{ strokeWidth: 2, r: 3 }}
                  activeDot={{ r: 8 }} 
                />
              )}
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
              {showInformer && (
                <Bar dataKey={PREDICTION_DATA_SERIES.INFORMER_MODEL} fill="#FF8C00" name="Informer Model" />
              )}
              {showGru && (
                <Bar dataKey={PREDICTION_DATA_SERIES.GRU_MODEL} fill="#8A2BE2" name="GRU Model" />
              )}
              {showCombined && (
                <Bar dataKey={PREDICTION_DATA_SERIES.PREDICTION_MODEL} fill="#0066CC" name="Combined Model" />
              )}
              {showActual && (
                <Bar dataKey={PREDICTION_DATA_SERIES.ACTUAL_PRICE} fill="#00CC66" name="Actual Price" />
              )}
            </BarChart>
          )}
        </ResponsiveContainer>
        <p className={styles.graphCaption}>
          Price prediction comparison for selected models
        </p>
      </div>
    </div>
  );
};

export default PredictionGraphs;