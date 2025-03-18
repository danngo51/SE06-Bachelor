declare module '*.geojson' {
    const value: string; // Treat the geojson import as a string (raw content)
    export default value;
}
