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
import interactiveAreaCodes from '../../utils/interactiveAreaCodes.ts'; // Import the interactive area codes
import FullMapMenu from '../FullMapMenu/FullMapMenu';

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


    const [loading, setLoading] = useState(false);

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
                                highlightedFeatureRef.current = feature;
                                highlightedLayerRef.current = layer;
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


    return (
        <>
            <LoadingSpinner loading={loading} />
            <div
                ref={mapContainerRef} // Map container reference
                style={{ height: '100vh', width: '100%' }} // Full screen map
            />
            {/* 
            <FullMapMenu
                handleMapFileUpload={handleMapFileUpload} 
                handleExcelFileUpload={handleExcelFileUpload}
                subdivide={subdivide}
                highlightBoolean={highlightBoolean}
                exportFeatureCollection={handleExportOfFiles}
                replaceAreaName={replaceAreaName}
                setAreaName={setNewAreaName}
            />
            */}
        </>
    );
};

export default FullMap;
