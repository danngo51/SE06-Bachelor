import L, { Map } from 'leaflet';

/**
 * Toggles GeoJSON layers on a Leaflet map.
 * @param geoJsonToRemove - The GeoJSON object to remove from the map.
 * @param geoJsonToAdd - The GeoJSON object to add to the map.
 * @param map - The Leaflet map instance.
 */
export function updateSubdivision(
    geoJsonToRemove: L.Layer | null,
    geoJsonToAdd: GeoJSON.GeoJSON | null,
    map: Map
): void {
    if (!map) {
        console.warn('Map instance is required.');
        return;
    }

    // Remove the GeoJSON layer if provided
    if (geoJsonToRemove) {
        map.removeLayer(geoJsonToRemove);
    }
}

export function generateChildKeys(key: string): string[] {
    return [`${key}.1`, `${key}.2`];
}
