import React, { useEffect, useState } from 'react';
import styles from './FullMapMenu.module.css';

// Define the props type
interface FullMapMenuProps {
    featureProb: (date: string) => void; // Modified to accept date parameter
    highlightBoolean: boolean;
    markedArea: GeoJSON.Feature | null; 
}

const FullMapMenu = ({
    featureProb,
    highlightBoolean,
    markedArea
}: FullMapMenuProps) => {
    const [selectedDate, setSelectedDate] = useState<string>('');
    
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
            featureProb(selectedDate);
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
