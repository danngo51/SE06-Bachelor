import { useState, useEffect } from 'react';
import { FeatureCollection } from 'geojson'; // Import GeoJSON types

const useUpdateGeoJSON = (geojsonPath: string) => {
    const [geojson, setGeojson] = useState<FeatureCollection | null>(null); // Store the GeoJSON data
    const [error, setError] = useState<string | null>(null); // Error state

    useEffect(() => {
        const fetchGeoJSON = async () => {
            try {
                const response = await fetch(geojsonPath); // Fetch the GeoJSON file
                if (!response.ok) {
                    throw new Error('Failed to fetch GeoJSON');
                }
                const data = await response.json();
                setGeojson(data); // Set the fetched GeoJSON data
            } catch (err) {
                setError('Failed to load GeoJSON' + err);
            }
        };

        fetchGeoJSON(); // Call the function to load the GeoJSON data
    }, [geojsonPath]); // Only re-run when the geojsonPath changes

    return { geojson, error };
};

export default useUpdateGeoJSON;
