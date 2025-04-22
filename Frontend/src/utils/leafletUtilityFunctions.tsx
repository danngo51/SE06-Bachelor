import { Dispatch, SetStateAction } from 'react';
import interactiveAreaCodes from './interactiveAreaCodes.ts'; // Import the interactive area codes
import L from 'leaflet';

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
        color: '#d6d6d6', // Grey border matching water color
        weight: 1,
        fillColor: interactiveAreaCodes.includes(feature.properties?.area) ? '#3388ff' : '#FFFFFF',
        fillOpacity: 1,
    };

    const fadedStyle = {
        color: '#d6d6d6', // Grey border matching water color
        weight: 1,
        fillColor: '#3388ff', // Blue but faded
        fillOpacity: 0.5,
    };

    const transparentStyle = {
        color: '#d6d6d6', // Grey border matching water color
        weight: 1,
        fillColor: '#FFFFFF',
        fillOpacity: 1,
    };

    const selectedStyle = {
        color: '#d6d6d6', // Grey border matching water color
        weight: 2, // Highlight border
        fillColor: 'green',
        fillOpacity: 1,
    };

    // Apply the initial style based on whether the area is interactive
    pathLayer.setStyle(normalStyle);

    // Disable interaction for non-interactive areas
    if (!interactiveAreaCodes.includes(feature.properties?.area)) {
        layer.off('click'); // Remove click event for non-interactive areas
        return;
    }

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
                    const featureLayer = layer as L.Path;
                    const featureArea = featureLayer.feature?.properties?.area;
                    featureLayer.setStyle(
                        interactiveAreaCodes.includes(featureArea) ? normalStyle : transparentStyle
                    );
                }
            });
        } else {
            // Reset all layers to faded or transparent style
            map.eachLayer((layer) => {
                if (layer instanceof L.Path && !drawnItems.hasLayer(layer)) {
                    const featureLayer = layer as L.Path;
                    const featureArea = featureLayer.feature?.properties?.area;
                    featureLayer.setStyle(
                        interactiveAreaCodes.includes(featureArea) ? fadedStyle : transparentStyle
                    );
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
    labelsLayerGroup: L.Map | L.LayerGroup<any>
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

export const removeMarkers = (geojson: L.GeoJSON) => {
    geojson.eachLayer((layer: L.Layer) => {
        if (layer instanceof L.Marker) {
            geojson.removeLayer(layer); // Remove markers from the GeoJSON layer
        }
    });
};
