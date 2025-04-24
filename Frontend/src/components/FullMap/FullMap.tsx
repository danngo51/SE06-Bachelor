import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import useUpdateGeoJSON from '../../hooks/useUpdateGeoJSON';
import {
    addInteractivityLayer,
    addLabelsLayer,
    styleGenerator,
    removeMarkers,
} from '../../utils/leafletUtilityFunctions';
import { initializeDrawingTools } from '../../utils/leafletUtilityDraw';
import FullMapMenu from '../FullMapMenu/FullMapMenu';
// Import from the selector utility instead of directly from the service files
import { fetchPredictionData, PredictionDataResponse } from '../../utils/apiSelector';
import PredictionGraphs from '../PredictionGraphs/PredictionGraphs';

import LoadingSpinner from '../LoadingSpinner/LoadingSpinner';

const FullMap = () => {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const geojsonPath = '/GeoJSON/worldMap.geojson'; // Path to your static GeoJSON file in public directory

    // First load static GeoJSON using useLoadGeoJSON hook
    const { geojson: staticGeoJSON } = useUpdateGeoJSON(geojsonPath);

    // To track the map and the current GeoJSON layer
    const mapRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null); // GeoJSON layer
    const geojsonLabelLayerRef = useRef<L.LayerGroup | null>(null); // GeoJSON label layer (LayerGroup)
    const drawingRef = React.useRef<L.FeatureGroup | null>(null);
    const highlightedFeatureRef = useRef<GeoJSON.Feature | null>(null);
    const highlightedLayerRef = useRef<L.Layer | null>(null);
    const [highlightBoolean, setHighlightBoolean] = useState(false);
    const [markedArea, setMarkedArea] = useState<GeoJSON.Feature | null>(null); // State to track the marked area as an object
    const [loading, setLoading] = useState(false);
    const [predictionData, setPredictionData] = useState<PredictionDataResponse | null>(null);
    const [showGraphs, setShowGraphs] = useState<boolean>(false);
    const [selectedDate, setSelectedDate] = useState<string>('');
    // New state variables to store the prediction info
    const [predictionCountry, setPredictionCountry] = useState<string>('');
    const [ setPredictionCountryCode] = useState<string>('');

    useEffect(() => {
        // Initialize the map only if mapContainerRef.current is not null
        if (mapContainerRef.current) {
            const map = L.map(mapContainerRef.current).setView(
                [55.000, 10.000],
                4
            ); // Set initial view
            mapRef.current = map;


            // Add the static GeoJSON data to the map
            if (staticGeoJSON) {
                geojsonLayerRef.current = L.geoJSON(staticGeoJSON, {
                    style: styleGenerator,
                    onEachFeature: (feature, layer) =>
                        addInteractivityLayer(
                            feature,
                            layer,
                            highlightedFeatureRef,
                            (feature, layer) => {
                                if (feature && feature.properties) {
                                    highlightedFeatureRef.current = feature;
                                    highlightedLayerRef.current = layer;
                                    setMarkedArea(feature); // Update marked area with the entire feature object
                                } else {
                                    setMarkedArea(null); // Handle null case, if nothing is marked
                                }
                            },
                            map,
                            drawingRef.current,
                            setHighlightBoolean
                        ),
                }).addTo(map);

                removeMarkers(geojsonLayerRef.current);

                geojsonLabelLayerRef.current = new L.LayerGroup();
                geojsonLabelLayerRef.current = addLabelsLayer(
                    staticGeoJSON,
                    geojsonLabelLayerRef.current
                );
                geojsonLabelLayerRef.current.addTo(map);
            }

            // Cleanup map on unmount
            return () => {
                map.remove();
            };
        }
    }, [staticGeoJSON]); // Re-run only when staticGeoJSON changes

    //Add drawing tools
    useEffect(() => {
        if (mapRef.current && !drawingRef.current) {
            const drawnItems = new L.FeatureGroup();
            drawingRef.current = drawnItems;

            // Initialize drawing tools only once
            initializeDrawingTools(mapRef.current, drawingRef.current);
        }
    }, []);

    useEffect(() => {
        if (markedArea !== null) {
            console.log('Marked area:', markedArea);
            console.log("area:", markedArea.properties?.area) // Log the entire object
        }
    }, [markedArea]); // Log to console whenever markedArea changes



    const featureProb = async (date: string) => {
        if (!markedArea || !markedArea.properties) {
            console.error('No country selected for prediction');
            return;
        }

        const countryName = markedArea.properties.country;
        const countryCode = markedArea.properties.iso_a2;
        
        // Check if countryCode exists, use a default if not
        if (!countryCode) {
            console.warn(`No ISO code found for ${countryName}, using 'XX' as fallback`);
        }
        
        console.log(`Prediction requested for ${countryName} (${countryCode || 'XX'}) on date: ${date}`);
        
        // Store the selected date and country for the prediction
        setSelectedDate(date);
        setPredictionCountry(countryName || 'Unknown Country');
        setPredictionCountryCode(countryCode || 'XX');
        
        try {
            console.log('Starting to fetch prediction data...');
            setLoading(true);
            
            // Call our prediction service to fetch the data, use 'XX' as fallback if no country code
            const data = await fetchPredictionData(countryCode || 'XX', date);
            console.log('Raw prediction data received:', data);
            
            // Check the data structure - updated to match our new data format
            if (!data.hourlyData || !data.predictionDate) {
                console.error('Received data has incorrect structure:', data);
                alert('Error: Prediction data has an invalid format');
                setLoading(false);
                return;
            }
            
            setPredictionData(data);
            setShowGraphs(true);
            console.log('Prediction data set to state, showGraphs set to true');
            
        } catch (error) {
            console.error('Error fetching prediction data:', error);
            alert('Error fetching prediction data. See console for details.');
        } finally {
            setLoading(false);
        }
    };
    
    // Function to close the graphs
    const handleCloseGraphs = () => {
        setShowGraphs(false);
    };

    // Add this for debugging
    useEffect(() => {
        if (predictionData) {
            console.log('=== DEBUG INFO ===');
            console.log('Prediction data state updated:', predictionData);
            console.log('showGraphs state:', showGraphs);
            console.log('selectedDate:', selectedDate);
            console.log('markedArea:', markedArea);
        }
    }, [predictionData, showGraphs, selectedDate, markedArea]);

    return (
        <>
            <LoadingSpinner loading={loading} />
            <div
                ref={mapContainerRef} // Map container reference
                style={{ height: '100vh', width: '100%' }} // Full screen map
            />
            
            <FullMapMenu
                featureProb={featureProb} 
                highlightBoolean={highlightBoolean}
                markedArea={markedArea} 
            />
            
            {/* Render prediction graphs based on showGraphs state, not on markedArea */}
            <PredictionGraphs
                data={predictionData}
                isVisible={showGraphs}
                onClose={handleCloseGraphs}
                countryName={predictionCountry}
                predictionDate={selectedDate}
            />
        </>
    );
};

export default FullMap;
