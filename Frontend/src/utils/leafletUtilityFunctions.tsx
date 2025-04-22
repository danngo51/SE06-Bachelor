import { Dispatch, SetStateAction } from 'react';
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
