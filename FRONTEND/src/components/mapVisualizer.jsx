import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, LayersControl } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const MapVisualization = ({ layersData }) => {
  const defaultPosition = [-0.4169, 36.9558]; // Nyeri, Kenya
  const defaultZoom = 10; // Initial zoom level

  // Initialize visibility state for each layer, excluding 'land-use-raster'
  const [visibleLayers, setVisibleLayers] = useState({});

  useEffect(() => {
    const initialVisibility = Object.keys(layersData).reduce((acc, key) => {
      if (key !== 'land-use-raster') {
        // Exclude 'land-use-raster'
        acc[key] = true;
      }
      return acc;
    }, {});
    setVisibleLayers(initialVisibility);
  }, [layersData]);

  const toggleLayer = (layerName) => {
    setVisibleLayers((prev) => ({ ...prev, [layerName]: !prev[layerName] }));
  };

  const layerStyles = {
    'area-of-interest': { color: 'orange', fillOpacity: 0.3 },
    soils: { color: 'brown', fillOpacity: 0.3 },
    geology: { color: 'gray', fillOpacity: 0.3 },
    digitalElevationModel: { color: 'black', fillOpacity: 0.1 },
    'protected-areas': { color: 'green', fillOpacity: 0.3 },
    rivers: { color: 'blue', weight: 2 },
    roads: { color: 'red', weight: 2 },
    settlement: { color: 'purple', fillOpacity: 0.5 },
    // Removed 'land-use-raster' styles
  };

  return (
    <div>
      {/* Layer Toggles */}
      <div className="layer-toggles mb-4">
        {Object.keys(layersData)
          .filter((layerName) => layerName !== 'land-use-raster')
          .map(
            (layerName) =>
              layersData[layerName] && (
                <label key={layerName} className="mr-4">
                  <input
                    type="checkbox"
                    checked={visibleLayers[layerName]}
                    onChange={() => toggleLayer(layerName)}
                    className="mr-1"
                  />
                  {layerName
                    .split('-')
                    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ')}
                </label>
              )
          )}
      </div>

      {/* Map Container */}
      <MapContainer
        center={defaultPosition}
        zoom={defaultZoom}
        style={{ height: '500px', width: '100%' }}
      >
        {/* Tile Layer */}
        <TileLayer
          attribution='&copy; <a href="https://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* GeoJSON Layers */}
        <LayersControl position="topright">
          {Object.entries(layersData)
            .filter(([layerName]) => layerName !== 'land-use-raster')
            .map(
              ([layerName, layerData]) =>
                visibleLayers[layerName] &&
                layerData &&
                layerData.features && (
                  <LayersControl.Overlay
                    checked
                    name={layerName
                      .split('-')
                      .map(
                        (word) => word.charAt(0).toUpperCase() + word.slice(1)
                      )
                      .join(' ')}
                    key={layerName}
                  >
                    <GeoJSON
                      data={layerData}
                      style={layerStyles[layerName]}
                      onEachFeature={(feature, layer) => {
                        layer.bindPopup(
                          `<pre>${JSON.stringify(
                            feature.properties,
                            null,
                            2
                          )}</pre>`
                        );
                      }}
                    />
                  </LayersControl.Overlay>
                )
            )}
        </LayersControl>
      </MapContainer>
    </div>
  );
};

export default MapVisualization;
