import React, { useEffect } from 'react';
import styles from './FullMapMenu.module.css';

// Define the props type
interface FullMapMenuProps {
    handleMapFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
    handleExcelFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
    subdivide: () => void;
    highlightBoolean: boolean;
    exportFeatureCollection: () => void;
    replaceAreaName: () => void;
    setAreaName: (name: string) => void;
}

const FullMapMenu = ({
    handleMapFileUpload,
    handleExcelFileUpload,
    subdivide,
    highlightBoolean,
    exportFeatureCollection,
    replaceAreaName,
    setAreaName,
}: FullMapMenuProps) => {
    useEffect(() => {}, [highlightBoolean]);

    const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        setAreaName(e.target.value); // Use e.target.value for the input's value
    };

    return (
        <>
            <div className={`${styles.fullMapMenuContainer}`}>
                <input
                    style={{ display: 'none' }}
                    id="mapInput"
                    type="file"
                    accept=".geojson"
                    onChange={handleMapFileUpload}
                    placeholder="Choose map"
                />
                <label className={`${styles.labelButton}`} htmlFor="mapInput">
                    Select map file
                </label>
                <input
                    style={{ display: 'none' }}
                    id="excelInput"
                    type="file"
                    accept=".xlsx, .xls"
                    onChange={handleExcelFileUpload}
                    placeholder="Choose excel file"
                />
                <label className={`${styles.labelButton}`} htmlFor="excelInput">
                    Select powerplants file
                </label>
                <button
                    onClick={subdivide}
                    disabled={!highlightBoolean} // Disable if array is empty
                    className={
                        highlightBoolean
                            ? `${styles.labelButton}`
                            : `${styles.labelButtonDisabled}`
                    }
                >
                    Subdivide
                </button>
                <button
                    onClick={exportFeatureCollection}
                    // Disable if array is empty
                    className={`${styles.labelButton}`}
                >
                    Export map and excel file
                </button>
                <div className={`${styles.areaNameContainer}`}>
                    <input
                        className={`${styles.customInput}`}
                        type="text"
                        onChange={handleInput}
                    />
                    <button
                        onClick={replaceAreaName}
                        disabled={!highlightBoolean} // Disable if array is empty
                        className={
                            highlightBoolean
                                ? `${styles.labelButton}`
                                : `${styles.labelButtonDisabled}`
                        }
                    >
                        Change area name
                    </button>
                </div>
            </div>
        </>
    );
};

export default FullMapMenu;
