import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import useUpdateGeoJSON from '../../hooks/useUpdateGeoJSON';
import useUploadGeoJSON from '../../hooks/useUploadGeoJSON';
import {
    addInteractivityLayer,
    addLabelsLayer,
    styleGenerator,
    getPriceAreaLabelCoordinates,
    removeLabelByAreaName,
    removeMarkers,
} from '../../utils/leafletUtilityFunctions';
import { initializeDrawingTools } from '../../utils/leafletUtilityDraw';
import { generateChildKeys } from '../../utils/leafletUtilitySubdivisionUpdate';
import FullMapMenu from '../FullMapMenu/FullMapMenu';
import {
    subdivideAreas,
    uploadExcelFile,
    createExcelFile,
    deletePowerPlants,
} from '../../api/api';
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

            const drawnItems = new L.FeatureGroup();
            drawingRef.current = initializeDrawingTools(
                mapRef.current,
                drawnItems
            );

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

    const subdivide = async () => {
        setLoading(true);

        const drawingGeoJSON = drawingRef.current?.toGeoJSON();
        const highlightedFeatureCollection = {
            type: 'FeatureCollection',
            features: [highlightedFeatureRef.current],
        };

        const dotIndex =
            highlightedFeatureRef.current?.properties.area.indexOf('.');
        const areaName = highlightedFeatureRef.current?.properties.area.slice(
            0,
            dotIndex
        );

        const subdivRequest = {
            AreaNames: [areaName],
            AddedPolygonFeatures: JSON.stringify(drawingGeoJSON),
            SelectedFeaturesString: JSON.stringify(
                highlightedFeatureCollection
            ),
        };

        const result = await subdivideAreas(subdivRequest); // Call the API function from api.js

        //Remove the highlighted layer from the main geojson layer which is used for exporting the map
        const newFeatures = result.features;
        console.log(newFeatures);
        console.log(typeof newFeatures);
        updateMainGeoJsonLayer(
            newFeatures,
            highlightedFeatureRef.current!.properties!.area
        );

        //Remove the highlighted feature. Add the subdivided features here:
        if (mapRef.current && highlightedLayerRef.current) {
            const map = mapRef.current;
            //Remove highlighted layer
            map.removeLayer(highlightedLayerRef.current);

            //Add subdivided labels
            const newLabels = generateChildKeys(
                highlightedFeatureRef.current?.properties.area
            );
            result.features[0].properties.area = newLabels[0];
            const test1 = getPriceAreaLabelCoordinates(
                result.features[0].geometry.coordinates
            );
            result.features[0].properties.priceAreaLabelCoordinates = [
                test1[0],
                test1[1],
            ];

            result.features[1].properties.area = newLabels[1];
            const test2 = getPriceAreaLabelCoordinates(
                result.features[1].geometry.coordinates
            );
            result.features[1].properties.priceAreaLabelCoordinates = [
                test2[0],
                test2[1],
            ];

            if (geojsonLabelLayerRef.current) {
                //Remove label from the previous area that has been subdivided
                removeLabelByAreaName(
                    geojsonLabelLayerRef.current,
                    highlightedFeatureRef.current?.properties.area
                );

                const newLabelLayer = addLabelsLayer(
                    result,
                    new L.LayerGroup()
                );

                geojsonLabelLayerRef.current.eachLayer((layer) => {
                    newLabelLayer.addLayer(layer);
                });

                geojsonLabelLayerRef.current = newLabelLayer;
                geojsonLabelLayerRef.current.addTo(map);
            }

            //Add interactivity to subdivided areas
            L.geoJSON(result, {
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

            //Remove drawing features
            drawingRef.current?.eachLayer((layer) => {
                map.removeLayer(layer); // Remove from map
                drawingRef.current?.removeLayer(layer); // Remove from the FeatureGroup
            });
        }

        setLoading(false);
    };

    //Helper method when subdividing for updating the main geojson layer used for exporting.
    const updateMainGeoJsonLayer = (
        newFeatures: GeoJSON.Feature[],
        areaName: string
    ) => {
        if (!geojsonLayerRef.current) return;

        const geoJsonLayer = geojsonLayerRef.current;
        // Update logic
        const updatedFeatures: GeoJSON.Feature[] = [];

        geoJsonLayer.eachLayer((layer) => {
            const feature = (layer as L.GeoJSON).feature as GeoJSON.Feature;

            // Check if the feature's `properties.area` matches any in `newFeatures`
            const hasMatchingArea = feature.properties!.area === areaName;

            if (hasMatchingArea) {
                // Remove the layer if there's a match
                geoJsonLayer.removeLayer(layer);
                newFeatures.forEach((newFeature) =>
                    updatedFeatures.push(newFeature)
                );
            } else {
                // Keep the rest of the features
                updatedFeatures.push(feature);
            }
        });

        // Clear the current GeoJSON layer
        geoJsonLayer.clearLayers();

        // Add the updated features back to the GeoJSON layer
        geoJsonLayer.addData({
            type: 'FeatureCollection',
            features: updatedFeatures,
        } as GeoJSON.FeatureCollection); // Explicit cast to FeatureCollection

        geojsonLayerRef.current = geoJsonLayer;
    };

    //Move this to seperate button
    const replaceAreaName = () => {
        const map = mapRef.current;

        if (
            map &&
            geojsonLabelLayerRef.current &&
            highlightedFeatureRef.current?.properties
        ) {
            //Remove label from the previous area that has been subdivided
            removeLabelByAreaName(
                geojsonLabelLayerRef.current,
                highlightedFeatureRef.current?.properties.area
            );

            const labelCoordinates =
                highlightedFeatureRef.current.properties
                    ?.priceAreaLabelCoordinates;

            if (newAreaName && labelCoordinates && newAreaName != 'none') {
                const label = L.divIcon({
                    className: 'map-area-name-label',
                    html: `<span class="map-area-name-label"> ${newAreaName} </span><br>`,
                    iconSize: [1, 1],
                    iconAnchor: [1, 1],
                });

                L.marker([labelCoordinates[1], labelCoordinates[0]], {
                    icon: label,
                }).addTo(geojsonLabelLayerRef.current);
            }

            geojsonLabelLayerRef.current.addTo(map);

            highlightedFeatureRef.current.properties.area = newAreaName;
        }
    };

    const handleExcelFileUpload = async (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        if (event.target.files && event.target.files.length > 0) {
            const excelFile = event.target.files[0];

            setLoading(true);
            try {
                await deletePowerPlants();
                await uploadExcelFile(excelFile);
                alert('File uploaded successfully!');
            } catch (error) {
                alert('File could not be uploaded: ' + error);
            } finally {
                setLoading(false);
            }
        }
    };

    const handleExportOfFiles = async () => {
        setLoading(true);

        const geojsonData = geojsonLayerRef.current?.toGeoJSON();
        const geoJSONString = JSON.stringify(geojsonData);
        const powerPlantRequest = { geoJSONString: geoJSONString };

        await createExcelFile(powerPlantRequest);

        setLoading(false);
    };

    return (
        <>
            <LoadingSpinner loading={loading} />
            <div
                ref={mapContainerRef} // Map container reference
                style={{ height: '100vh', width: '100%' }} // Full screen map
            />
            <FullMapMenu
                handleMapFileUpload={handleMapFileUpload}
                handleExcelFileUpload={handleExcelFileUpload}
                subdivide={subdivide}
                highlightBoolean={highlightBoolean}
                exportFeatureCollection={handleExportOfFiles}
                replaceAreaName={replaceAreaName}
                setAreaName={setNewAreaName}
            />
        </>
    );
};

export default FullMap;
