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
    // Store weights as percentages for display (0-100 range)
    const [weightsPercentage, setWeightsPercentage] = useState({ gru: 0, informer: 0, xgboost: 100 });
    
    useEffect(() => {
        // Reset selected date when highlight state changes
        setSelectedDate('');
    }, [highlightBoolean]);

    // Function to handle date change
    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedDate(e.target.value);
    };    // Convert percentage weights to decimal weights for API
    const convertToDecimalWeights = (percentageWeights: { gru: number; informer: number; xgboost: number }) => {
        return {
            gru: percentageWeights.gru / 100,
            informer: percentageWeights.informer / 100,
            xgboost: percentageWeights.xgboost / 100
        };
    };

    // Function to handle weight changes with validation
    const handleWeightChange = (model: 'gru' | 'informer' | 'xgboost', newValue: number) => {
        // Ensure the new value is not negative
        const clampedValue = Math.max(0, newValue);
        
        // Calculate current total excluding the model being changed
        const otherModelsTotal = Object.keys(weightsPercentage)
            .filter(key => key !== model)
            .reduce((sum, key) => sum + weightsPercentage[key as keyof typeof weightsPercentage], 0);
        
        // Ensure the new value doesn't make total exceed 100
        const maxAllowedValue = 100 - otherModelsTotal;
        const finalValue = Math.min(clampedValue, maxAllowedValue);
        
        setWeightsPercentage({
            ...weightsPercentage,
            [model]: finalValue
        });
    };    // Function to auto-balance weights to total 100%
    const autoBalanceWeights = () => {
        const currentTotal = totalWeight;
        if (currentTotal === 0) {
            // If all weights are 0, distribute equally
            setWeightsPercentage({ gru: 33, informer: 33, xgboost: 34 });
        } else if (currentTotal !== 100) {
            // Proportionally adjust all weights to sum to 100
            const scaleFactor = 100 / currentTotal;
            let newGru = Math.round(weightsPercentage.gru * scaleFactor);
            let newInformer = Math.round(weightsPercentage.informer * scaleFactor);
            let newXgboost = Math.round(weightsPercentage.xgboost * scaleFactor);
            
            // Handle rounding errors to ensure total is exactly 100
            const newTotal = newGru + newInformer + newXgboost;
            const diff = 100 - newTotal;
            
            // Adjust the largest weight to account for rounding differences
            if (diff !== 0) {
                if (newXgboost >= newGru && newXgboost >= newInformer) {
                    newXgboost += diff;
                } else if (newGru >= newInformer) {
                    newGru += diff;
                } else {
                    newInformer += diff;
                }
            }
            
            setWeightsPercentage({
                gru: Math.max(0, newGru),
                informer: Math.max(0, newInformer),
                xgboost: Math.max(0, newXgboost)
            });
        }
    };

    // Calculate if total weights equal 100%
    const totalWeight = weightsPercentage.gru + weightsPercentage.informer + weightsPercentage.xgboost;
    const isValidWeightTotal = totalWeight === 100;

    // Function to handle predict button click
    const handlePredict = () => {
        if (selectedDate && isValidWeightTotal) {
            const decimalWeights = convertToDecimalWeights(weightsPercentage);
            featureProb(selectedDate, decimalWeights);
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
                        </div>                        <div className={styles.weightsContainer}>
                            <label className={styles.weightsLabel}>Model Weights (%):</label>
                            <div className={styles.weightInputs}>
                                <div className={styles.weightInput}>
                                    <label htmlFor="gru-weight">GRU:</label>
                                    <input 
                                        type="number" 
                                        id="gru-weight"
                                        min="0" 
                                        max="100" 
                                        step="1"
                                        value={weightsPercentage.gru}
                                        onChange={(e) => handleWeightChange('gru', parseInt(e.target.value) || 0)}
                                    />
                                    <span>%</span>
                                </div>
                                <div className={styles.weightInput}>
                                    <label htmlFor="informer-weight">Informer:</label>
                                    <input 
                                        type="number" 
                                        id="informer-weight"
                                        min="0" 
                                        max="100" 
                                        step="1"
                                        value={weightsPercentage.informer}
                                        onChange={(e) => handleWeightChange('informer', parseInt(e.target.value) || 0)}
                                    />
                                    <span>%</span>
                                </div>
                                <div className={styles.weightInput}>
                                    <label htmlFor="xgboost-weight">XGBoost:</label>
                                    <input 
                                        type="number" 
                                        id="xgboost-weight"
                                        min="0" 
                                        max="100" 
                                        step="1"
                                        value={weightsPercentage.xgboost}
                                        onChange={(e) => handleWeightChange('xgboost', parseInt(e.target.value) || 0)}
                                    />
                                    <span>%</span>
                                </div>
                            </div>                            <div className={`${styles.totalWeight} ${!isValidWeightTotal ? styles.totalWeightError : styles.totalWeightValid}`}>
                                Total: {totalWeight}% {!isValidWeightTotal && '(Must equal 100%)'}
                            </div>
                            {!isValidWeightTotal && (
                                <button 
                                    type="button"
                                    onClick={autoBalanceWeights}
                                    className={styles.autoBalanceButton}
                                >
                                    Auto Balance to 100%
                                </button>
                            )}
                        </div>
                          <button
                            onClick={handlePredict}
                            className={(selectedDate && isValidWeightTotal) ? `${styles.labelButton}` : `${styles.labelButtonDisabled}`}
                            disabled={!selectedDate || !isValidWeightTotal}
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
