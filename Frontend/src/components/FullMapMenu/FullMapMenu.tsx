import React, { useEffect, useState } from 'react';
import styles from './FullMapMenu.module.css';

// Define the props type
interface FullMapMenuProps {
    featureProb: (date: string, weights?: { gru: number; informer: number; xgboost: number }) => void;
    highlightBoolean: boolean;
    markedArea: GeoJSON.Feature | null; 
}

const FullMapMenu = ({
    featureProb,
    highlightBoolean,
    markedArea
}: FullMapMenuProps) => {
    const [selectedDate, setSelectedDate] = useState<string>('');
    const [weights, setWeights] = useState({ gru: 0.0, informer: 0.0, xgboost: 1.0 });
    
    useEffect(() => {
        // Reset selected date when highlight state changes
        setSelectedDate('');
    }, [highlightBoolean]);

    // Function to handle date change
    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedDate(e.target.value);
    };
      // Function to handle predict button click
    const handlePredict = () => {
        if (selectedDate) {
            featureProb(selectedDate, weights);
        }
    };
    
    return (
        <>
            <div className={`${styles.fullMapMenuContainer}`}>
                {!highlightBoolean ? (
                    // Show only "Select a country" label when no area is selected
                    <div className={`${styles.areaNameContainer}`}>
                        <label className={`${styles.areaNameLabel}`}>
                            Select a country
                        </label>
                    </div>
                ) : (
                    // Show area name, date picker, and predict button when an area is selected
                    <div className={`${styles.areaNameContainer} ${styles.expandedContainer}`}>
                        <label className={`${styles.areaNameLabel}`}> 
                            {markedArea ? markedArea.properties?.country : null}
                        </label>
                          <div className={styles.datePickerContainer}>
                            <label htmlFor="prediction-date" className={styles.dateLabel}>
                                Select a date to be predicted:
                            </label>
                            <input 
                                type="date" 
                                id="prediction-date"
                                className={styles.datePicker}
                                value={selectedDate}
                                onChange={handleDateChange}
                            />
                        </div>
                        
                        <div className={styles.weightsContainer}>
                            <label className={styles.weightsLabel}>Model Weights:</label>
                            <div className={styles.weightInputs}>
                                <div className={styles.weightInput}>
                                    <label htmlFor="gru-weight">GRU:</label>
                                    <input 
                                        type="number" 
                                        id="gru-weight"
                                        min="0" 
                                        max="1" 
                                        step="0.1"
                                        value={weights.gru}
                                        onChange={(e) => setWeights({...weights, gru: parseFloat(e.target.value) || 0})}
                                    />
                                </div>
                                <div className={styles.weightInput}>
                                    <label htmlFor="informer-weight">Informer:</label>
                                    <input 
                                        type="number" 
                                        id="informer-weight"
                                        min="0" 
                                        max="1" 
                                        step="0.1"
                                        value={weights.informer}
                                        onChange={(e) => setWeights({...weights, informer: parseFloat(e.target.value) || 0})}
                                    />
                                </div>
                                <div className={styles.weightInput}>
                                    <label htmlFor="xgboost-weight">XGBoost:</label>
                                    <input 
                                        type="number" 
                                        id="xgboost-weight"
                                        min="0" 
                                        max="1" 
                                        step="0.1"
                                        value={weights.xgboost}
                                        onChange={(e) => setWeights({...weights, xgboost: parseFloat(e.target.value) || 0})}
                                    />
                                </div>
                            </div>
                        </div>
                        
                        <button
                            onClick={handlePredict} // Changed to use handlePredict
                            className={selectedDate ? `${styles.labelButton}` : `${styles.labelButtonDisabled}`}
                            disabled={!selectedDate}
                        >
                            Predict
                        </button>
                    </div>
                )}
            </div>
        </>
    );
};

export default FullMapMenu;
