import React from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const MapVisualization = ({ geoJsonData }) => {
  const defaultPosition = [51.505, -0.09]; // Default center (e.g., London)

  return (
    <MapContainer
      center={defaultPosition}
      zoom={6}
      style={{ height: '500px', width: '100%' }}
    >
      <TileLayer
        attribution='&copy; <a href="https://osm.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {geoJsonData && geoJsonData.features && <GeoJSON data={geoJsonData} />}
    </MapContainer>
  );
};

export default MapVisualization;
