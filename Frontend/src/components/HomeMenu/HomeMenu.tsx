import React, { useState } from 'react';
import styles from './HomeMenu.module.css';
// import e2gLogo from '../../assets/images/e2g-logo-hvid-menu.png'
import strategiLogo from '../../assets/images/strategi-rum.png';
import lightningSymbol from '../../assets/svg/lightning.svg';
import { Link } from 'react-router-dom';
import TransmissionDialog from '../Dialog/TransmissionDialog/TransmissionDialog';
import PredictDialog from '../Dialog/PredictDialog/PredictDialog';

// Define the structure of menu items
interface MenuItem {
    id: number;
    label: string;
    path?: string; // Optional path for navigation
    action?: () => void; // Optional action for non-navigation functionality
}

const HomeMenu = () => {
    const [visibleDialogId, setVisibleDialogId] = useState<number | null>(null);

    const menuItems: MenuItem[] = [
        { id: 1, label: 'Predict' },
        { id: 2, label: 'Indlæs produktionsprofiler' },
    ];

    const handleDialogOpen = (id: number) => {
        setVisibleDialogId(id);
    };

    const handleDialogClose = () => {
        setVisibleDialogId(null);
    };

    return (
        <div className={`${styles.homeMenuContainer}`}>
            <div className={`${styles.homeMenu}`}>
                <div className={`${styles.homeTitleContainer}`}>
                    <img src={strategiLogo} alt="" />
                    {/* <h2 className={`${styles.menuTitle}`}>Indlæsningsmodel</h2> */}
                </div>

                {menuItems.map((item) => (
                    <div key={item.id} className={`${styles.menuItem}`}>
                        {/* Check if the item has a 'path' for navigation */}
                        {item.path ? (
                            <Link
                                to={item.path}
                                className={`${styles.menuLink}`}
                            >
                                <span className={`${styles.menuItemText}`}>
                                    {item.label}
                                </span>
                                <img
                                    className={`${styles.svgItem}`}
                                    src={lightningSymbol}
                                    alt="Icon"
                                />
                            </Link>
                        ) : (
                            // If there's no 'path', check if there's an 'action' function
                            <button
                                onClick={() => handleDialogOpen(item.id)}
                                className={`${styles.menuInvisibleButton}`}
                            >
                                <span className={`${styles.menuItemText}`}>
                                    {item.label}
                                </span>
                                <img
                                    className={`${styles.svgItem}`}
                                    src={lightningSymbol}
                                    alt="Icon"
                                />
                            </button>
                        )}
                    </div>
                ))}
            </div>

        
            <TransmissionDialog
                visible={visibleDialogId === 2}
                onClose={handleDialogClose}
            />
            <PredictDialog
                visible={visibleDialogId === 1}
                onClose={handleDialogClose}
            />
        </div>
    );
};

export default HomeMenu;
