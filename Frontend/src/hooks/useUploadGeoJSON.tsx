import { useState } from 'react';
import * as GeoJSON from 'geojson'; // Import GeoJSON types

const useUploadGeoJSON = () => {
    const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(
        null
    );
    const [error, setError] = useState<string | null>(null); // Error state

    // Function to handle file upload
    const handleMapFileUpload = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const file = event.target.files?.[0]; // Get the first file
        if (!file) {
            setError('No file selected');
            return;
        }

        if (
            file.type !== 'application/geo+json' &&
            !file.name.endsWith('.geojson')
        ) {
            setError('Please select a valid GeoJSON file');
            return;
        }

        const reader = new FileReader();
        setError(null); // Reset any previous errors

        reader.onload = () => {
            try {
                const parsedGeoJSON = JSON.parse(reader.result as string); // Parse the JSON content
                setGeojson(parsedGeoJSON); // Set the GeoJSON data
            } catch (err) {
                setError('Failed to parse GeoJSON' + err); // Handle invalid JSON
            }
        };

        reader.onerror = () => {
            setError('Error reading the file'); // Handle file read error
        };

        reader.readAsText(file); // Read the file as a text string
    };

    return { geojson, error, handleMapFileUpload };
};

export default useUploadGeoJSON;
