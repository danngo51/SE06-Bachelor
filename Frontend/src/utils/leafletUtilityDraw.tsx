import L, { Map, FeatureGroup } from 'leaflet';
import 'leaflet-draw';
import 'leaflet-draw/dist/leaflet.draw.css';

export const initializeDrawingTools = (
    map: Map,
    drawnItems: L.FeatureGroup
): FeatureGroup => {
    map.addLayer(drawnItems);
    const drawControl = new L.Control.Draw({
        position: 'topleft',
        edit: {
            featureGroup: drawnItems,
        },
        draw: {
            circlemarker: false,
            polyline: false,
            marker: false,
            rectangle: false,
            circle: false,
        },
    });

    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (event) => {
        const layer = event.layer;
        drawnItems.addLayer(layer);
    });

    return drawnItems;
};
