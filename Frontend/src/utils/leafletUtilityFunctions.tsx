import { Dispatch, SetStateAction } from 'react';
import L from 'leaflet';
import { GeoJSON, Feature, FeatureCollection } from 'geojson';

export const addInteractivityLayer = (
    feature: GeoJSON.Feature,
    layer: L.Layer,
    highlightedFeatureRef: React.MutableRefObject<GeoJSON.Feature | null>,
    setHighlightedFeature: (feature: GeoJSON.Feature, layer: L.Layer) => void,
    map: L.Map,
    drawnItems: L.FeatureGroup,
    setHighlightBoolean: Dispatch<SetStateAction<boolean>>
) => {
    const pathLayer = layer as L.Path;

    const normalStyle = {
        color: 'white',
        weight: 1,
        fillColor: '#3388ff',
        fillOpacity: 1,
    };

    const transparentStyle = {
        color: '#d6d6d6',
        weight: 1,
        fillColor: '#FFFFFF',
        fillOpacity: 1,
    };

    const selectedStyle = {
        color: 'white',
        weight: 2, // Highlight border
        fillColor: 'green',
        fillOpacity: 1,
    };

    // Click event handler
    layer.on('click', () => {
        const highlightedFeature = highlightedFeatureRef.current; // Always use the latest ref value

        if (highlightedFeature === feature) {
            // Unselect current feature
            pathLayer.setStyle(normalStyle);
            setHighlightedFeature(null, null);
            setHighlightBoolean(false);

            map.eachLayer((layer) => {
                if (layer instanceof L.Path && !drawnItems.hasLayer(layer)) {
                    layer.setStyle(normalStyle);
                }
            });
        } else {
            // Reset all layers to normal style
            map.eachLayer((layer) => {
                if (layer instanceof L.Path && !drawnItems.hasLayer(layer)) {
                    layer.setStyle(transparentStyle);
                }
            });

            // Apply selected style to the clicked feature
            pathLayer.setStyle(selectedStyle);
            setHighlightedFeature(feature, layer);
            setHighlightBoolean(true);
        }
    });
};

export const styleGenerator = (): L.PathOptions => ({
    color: 'white',
    weight: 1,
    fillColor: '#3388ff',
    fillOpacity: 1,
});

// Utility function to add labels
export const addLabelsLayer = (
    geoJsonData: GeoJSON.GeoJsonObject,
    labelsLayerGroup
): L.LayerGroup => {
    labelsLayerGroup = L.layerGroup();

    L.geoJSON(geoJsonData, {
        onEachFeature: (feature) => {
            const areaName = feature.properties?.area;
            const labelCoordinates =
                feature.properties?.priceAreaLabelCoordinates;

            if (areaName && labelCoordinates && areaName != 'none') {
                const label = L.divIcon({
                    className: 'map-area-name-label',
                    html: `<span class="map-area-name-label"> ${areaName} </span><br>`,
                    iconSize: [1, 1],
                    iconAnchor: [1, 1],
                });

                L.marker([labelCoordinates[1], labelCoordinates[0]], {
                    icon: label,
                }).addTo(labelsLayerGroup);
            }
        },
    });

    return labelsLayerGroup;
};

export const removeLabelByAreaName = (
    labelsLayerGroup: L.LayerGroup,
    areaName: string
): void => {
    labelsLayerGroup.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
            const icon = layer.options.icon;
            if (icon instanceof L.DivIcon) {
                const labelText = icon.options.html;

                // Use regex to match the exact area name
                const regex = new RegExp(`\\b${areaName}\\b`); // Matches `areaName` as a whole word
                if (regex.test(labelText)) {
                    labelsLayerGroup.removeLayer(layer); // Remove the marker (label) from the layer group
                }
            }
        }
    });
};

export const getPriceAreaLabelCoordinates = (
    nestedArray: number[][][][]
): [number, number] => {
    // Step 1: Flatten the deeply nested array to a single array of numbers
    const flattened = nestedArray.flat(3).flat();

    // Step 2: Calculate the averages for the first and second numbers
    const total = flattened.reduce(
        (acc, num, index) => {
            if (index % 2 === 0) {
                acc[0] += num; // Sum of all first numbers (even index)
            } else {
                acc[1] += num; // Sum of all second numbers (odd index)
            }
            return acc;
        },
        [0, 0] // Initial values for [x, y]
    );

    // Step 3: Calculate and return averages
    const length = flattened.length / 2; // Dividing by 2 since we have pairs
    return [total[0] / length, total[1] / length];
};

// Type guard to check if a layer can be converted to GeoJSON
export const isGeoJSONLayer = (
    layer: L.Layer
): layer is
    | L.Marker
    | L.Polygon
    | L.Polyline
    | L.Circle
    | L.Rectangle
    | L.GeoJSON => {
    return (
        typeof (
            layer as L.Polygon | L.Polyline | L.Circle | L.Rectangle | L.GeoJSON
        ).toGeoJSON === 'function'
    );
};

// Function to extract all features from a map and return them as a FeatureCollection
export const extractFeatureCollection = (map: L.Map) => {
    if (!map) return;

    const features: Feature[] = [];

    map.eachLayer((layer: L.Layer) => {
        if (layer instanceof L.Marker) return; // Skip markers

        // Only process layers that support the toGeoJSON method
        if (isGeoJSONLayer(layer)) {
            try {
                const feature = layer.toGeoJSON() as Feature;

                features.push(feature);
            } catch (err) {
                console.error('Error converting layer to GeoJSON:', err);
            }
        }
    });

    const featureCollection: FeatureCollection = {
        type: 'FeatureCollection',
        features,
    };

    exportToFile(featureCollection);
};

// Function to export the FeatureCollection to a .txt file
const exportToFile = (featureCollection: FeatureCollection) => {
    const jsonContent = JSON.stringify(featureCollection, null, 2); // Convert the object to JSON string
    const blob = new Blob([jsonContent], { type: 'application/json' }); // Create a Blob from the JSON string
    const url = URL.createObjectURL(blob); // Create an object URL for the Blob

    // Create an anchor element for downloading
    const a = document.createElement('a');
    a.href = url;
    a.download = 'features.geojson'; // The filename to be used when downloading
    document.body.appendChild(a); // Append the anchor to the document body
    a.click(); // Trigger the download
    document.body.removeChild(a); // Clean up the DOM by removing the anchor element
};

export const removeMarkers = (geojson: L.GeoJSON) => {
    geojson.eachLayer((layer: L.Layer) => {
        if (layer instanceof L.Marker) {
            geojson.removeLayer(layer); // Remove markers from the GeoJSON layer
        }
    });
};
