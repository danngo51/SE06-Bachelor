import React, { useEffect } from 'react';
import styles from './FullMapMenu.module.css';

// Define the props type
interface FullMapMenuProps {
    featureProb: () => void;
    highlightBoolean: boolean;
}

const FullMapMenu = ({
    featureProb,
    
    highlightBoolean,
    
}: FullMapMenuProps) => {
    useEffect(() => {}, [highlightBoolean]);

    
    return (
        <>
            <div className={`${styles.fullMapMenuContainer}`}>
                <div className={`${styles.areaNameContainer}`}>
                    <button
                        onClick={featureProb}
                        disabled={!highlightBoolean} // Disable if array is empty
                        className={
                            highlightBoolean
                                ? `${styles.labelButton}`
                                : `${styles.labelButtonDisabled}`
                        }
                    >
                        Predict
                    </button>
                </div>
            </div>
        </>
    );
};

export default FullMapMenu;
