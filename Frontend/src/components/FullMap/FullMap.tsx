import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import useUpdateGeoJSON from '../../hooks/useUpdateGeoJSON';
import useUploadGeoJSON from '../../hooks/useUploadGeoJSON';
import {
    addInteractivityLayer,
    addLabelsLayer,
    styleGenerator,
    removeMarkers,
} from '../../utils/leafletUtilityFunctions';
import { initializeDrawingTools } from '../../utils/leafletUtilityDraw';
import FullMapMenu from '../FullMapMenu/FullMapMenu';

import LoadingSpinner from '../LoadingSpinner/LoadingSpinner';

const FullMap = () => {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const geojsonPath = '/GeoJSON/worldMap.geojson'; // Path to your static GeoJSON file in public directory

    // First load static GeoJSON using useLoadGeoJSON hook
    const { geojson: staticGeoJSON } = useUpdateGeoJSON(geojsonPath);

    // Handle changing of geojson map
    const { geojson: uploadedGeoJSON, handleMapFileUpload } =
        useUploadGeoJSON();

    // To track the map and the current GeoJSON layer
    const mapRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null); // GeoJSON layer
    const geojsonLabelLayerRef = useRef<L.LayerGroup | null>(null); // GeoJSON label layer (LayerGroup)
    const drawingRef = React.useRef<L.FeatureGroup | null>(null);
    const highlightedFeatureRef = useRef<GeoJSON.Feature | null>(null);
    const highlightedLayerRef = useRef<L.Layer | null>(null);
    const [highlightBoolean, setHighlightBoolean] = useState(false);

    const [newAreaName, setNewAreaName] = useState('');

    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // Initialize the map only if mapContainerRef.current is not null
        if (mapContainerRef.current) {
            const map = L.map(mapContainerRef.current).setView(
                [51.505, -0.09],
                3
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

    useEffect(() => {
        // Check if the uploaded GeoJSON is present and map is initialized
        if (uploadedGeoJSON && mapRef.current) {
            const map = mapRef.current;

            // Remove the GeoJSON layer if it exists
            if (geojsonLayerRef.current) {
                map.removeLayer(geojsonLayerRef.current);
                geojsonLayerRef.current = null; // Reset the reference
            }

            //Remove labels
            if (geojsonLabelLayerRef.current) {
                map.removeLayer(geojsonLabelLayerRef.current);
                geojsonLabelLayerRef.current = null; // Reset the reference
            }

            // Add the uploaded GeoJSON layer to the map
            geojsonLayerRef.current = L.geoJSON(uploadedGeoJSON, {
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

            // Iterate through the layers in the GeoJSON layer and remove points (markers)
            removeMarkers(geojsonLayerRef.current);

            //Re add labels
            geojsonLabelLayerRef.current = new L.LayerGroup();
            geojsonLabelLayerRef.current = addLabelsLayer(
                uploadedGeoJSON,
                geojsonLabelLayerRef.current
            );
            geojsonLabelLayerRef.current.addTo(map);
        }
    }, [uploadedGeoJSON]); // Re-run whenever the uploadedGeoJSON changes

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
