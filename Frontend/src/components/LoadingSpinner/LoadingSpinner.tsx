import React from 'react';
import { ProgressSpinner } from 'primereact/progressspinner';
import './LoadingSpinner.css'; // Import the CSS for styling

interface LoadingSpinnerProps {
    loading: boolean;
}

const LoadingSpinner = ({ loading }: LoadingSpinnerProps) => {
    if (!loading) return null;

    return (
        <div className="loading-overlay">
            <div>
                <ProgressSpinner className="loading-spinner" />
            </div>
        </div>
    );
};

export default LoadingSpinner;
