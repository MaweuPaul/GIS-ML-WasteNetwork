import React, { useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// This component will handle updating the map view when geoJsonData changes
const MapUpdater = ({ geoJsonData }) => {
  const map = useMap();

  useEffect(() => {
    if (geoJsonData && geoJsonData.features.length > 0) {
      const geoJsonLayer = L.geoJSON(geoJsonData);
      const bounds = geoJsonLayer.getBounds();
      map.fitBounds(bounds);
    }
  }, [geoJsonData, map]);

  return null;
};

const getRandomColor = () => {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

const MapVisualization = ({ geoJsonData }) => {
  if (!geoJsonData || geoJsonData.features.length === 0) {
    return (
      <div
        style={{
          height: '400px',
          width: '100%',
          backgroundColor: '#f0f0f0',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <p>Upload data to visualize on the map</p>
      </div>
    );
  }

  const style = () => {
    return { color: getRandomColor() };
  };

  return (
    <MapContainer
      style={{ height: '400px', width: '100%' }}
      center={[0, 0]}
      zoom={2}
      scrollWheelZoom={false}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <GeoJSON data={geoJsonData} style={style} />
      <MapUpdater geoJsonData={geoJsonData} />
    </MapContainer>
  );
};

export default MapVisualization;
