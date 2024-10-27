import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  LayersControl,
  useMap,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import debounce from 'lodash/debounce';

const FeatureHighlighter = ({ feature, popupContent }) => {
  const map = useMap();

  useEffect(() => {
    if (feature && feature.geometry) {
      const bounds = L.geoJSON(feature).getBounds();
      map.fitBounds(bounds, { padding: [50, 50] });

      if (popupContent) {
        L.popup()
          .setLatLng(bounds.getCenter())
          .setContent(popupContent)
          .openOn(map);
      }
    }
  }, [feature, map, popupContent]);

  return null;
};

const HighlightedText = React.memo(({ text, highlight }) => {
  if (!highlight.trim()) {
    return <span>{text}</span>;
  }
  const regex = new RegExp(`(${highlight})`, 'gi');
  const parts = text.split(regex);
  return (
    <span>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i}>{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  );
});

const MapVisualization = ({ layersData }) => {
  const defaultPosition = [-0.4169, 36.9558]; // Nyeri, Kenya
  const defaultZoom = 10; // Initial zoom level

  const [visibleLayers, setVisibleLayers] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [selectedLayerType, setSelectedLayerType] = useState('all');
  const [popupContent, setPopupContent] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const resultsPerPage = 10;

  useEffect(() => {
    const initialVisibility = Object.keys(layersData).reduce((acc, key) => {
      if (key !== 'land-use-raster') {
        acc[key] = true;
      }
      return acc;
    }, {});
    setVisibleLayers(initialVisibility);
  }, [layersData]);

  const toggleLayer = useCallback((layerName) => {
    setVisibleLayers((prev) => ({ ...prev, [layerName]: !prev[layerName] }));
  }, []);

  const layerStyles = useMemo(
    () => ({
      'area-of-interest': { color: 'orange', fillOpacity: 0.3 },
      soils: { color: 'brown', fillOpacity: 0.3 },
      geology: { color: 'gray', fillOpacity: 0.3 },
      digitalElevationModel: { color: 'black', fillOpacity: 0.1 },
      'protected-areas': { color: 'green', fillOpacity: 0.3 },
      rivers: { color: 'blue', weight: 2 },
      roads: { color: 'red', weight: 2 },
      settlement: { color: 'purple', fillOpacity: 0.5 },
    }),
    []
  );

  const debouncedSearch = useMemo(
    () =>
      debounce((term) => {
        setIsLoading(true);
        setError(null);
        try {
          const results = [];
          Object.entries(layersData).forEach(([layerName, layerData]) => {
            if (layerData && Array.isArray(layerData.features)) {
              layerData.features.forEach((feature) => {
                if (feature && feature.properties) {
                  const properties = Object.values(feature.properties);
                  const match = properties.some((value) =>
                    String(value).toLowerCase().includes(term.toLowerCase())
                  );
                  if (match) {
                    results.push({ layerName, feature });
                  }
                }
              });
            }
          });
          setSearchResults(results);
        } catch (err) {
          setError('An error occurred while searching. Please try again.');
        } finally {
          setIsLoading(false);
        }
      }, 300),
    [layersData]
  );

  useEffect(() => {
    if (searchTerm) {
      debouncedSearch(searchTerm);
    } else {
      setSearchResults([]);
    }
    setCurrentPage(1);
  }, [searchTerm, debouncedSearch]);

  const handleResultClick = useCallback((result) => {
    setSelectedFeature(result.feature);
    const content = `
      <div style="max-width: 300px;">
        <h3 style="font-weight: bold; margin-bottom: 8px;">${
          result.feature.properties.name || 'Feature Details'
        }</h3>
        <table style="width: 100%; border-collapse: collapse;">
          ${Object.entries(result.feature.properties)
            .map(
              ([key, value]) => `
              <tr>
                <td style="font-weight: bold; padding: 4px; border: 1px solid #ddd;">${key}</td>
                <td style="padding: 4px; border: 1px solid #ddd;">${value}</td>
              </tr>
            `
            )
            .join('')}
        </table>
      </div>
    `;
    setPopupContent(content);
  }, []);

  const filteredResults = useMemo(
    () =>
      selectedLayerType === 'all'
        ? searchResults
        : searchResults.filter(
            (result) => result.layerName === selectedLayerType
          ),
    [searchResults, selectedLayerType]
  );

  const paginatedResults = useMemo(() => {
    const startIndex = (currentPage - 1) * resultsPerPage;
    return filteredResults.slice(startIndex, startIndex + resultsPerPage);
  }, [filteredResults, currentPage]);

  const totalPages = Math.ceil(filteredResults.length / resultsPerPage);

  const clearSearch = () => {
    setSearchTerm('');
    setSearchResults([]);
    setCurrentPage(1);
  };

  return (
    <div>
      <div className="flex mb-4">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search features..."
          className="flex-grow px-3 py-2 border border-gray-300 rounded-md mr-2"
        />
        <button
          onClick={clearSearch}
          className="px-3 py-2 bg-gray-200 rounded-md mr-2"
        >
          Clear
        </button>
        <select
          value={selectedLayerType}
          onChange={(e) => setSelectedLayerType(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="all">All Layers</option>
          {Object.keys(layersData).map((layerName) => (
            <option key={layerName} value={layerName}>
              {layerName}
            </option>
          ))}
        </select>
      </div>

      {/* Search Results */}
      {isLoading && <div className="mb-4">Loading...</div>}
      {error && <div className="mb-4 text-red-500">{error}</div>}
      {!isLoading && !error && filteredResults.length > 0 && (
        <div className="mb-4 max-h-60 overflow-y-auto border border-gray-300 rounded-md p-2">
          <h3 className="font-bold mb-2">
            Search Results: {filteredResults.length} found
          </h3>
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left">Layer</th>
                <th className="text-left">Properties</th>
              </tr>
            </thead>
            <tbody>
              {paginatedResults.map((result, index) => (
                <tr
                  key={index}
                  className="cursor-pointer hover:bg-gray-100"
                  onClick={() => handleResultClick(result)}
                >
                  <td className="pr-2">{result.layerName}</td>
                  <td>
                    {Object.entries(result.feature.properties).map(
                      ([key, value]) => (
                        <div key={key}>
                          <strong>{key}:</strong>{' '}
                          <HighlightedText
                            text={String(value)}
                            highlight={searchTerm}
                          />
                        </div>
                      )
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {totalPages > 1 && (
            <div className="mt-4 flex justify-between items-center">
              <button
                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="px-2 py-1 bg-gray-200 rounded-md disabled:opacity-50"
              >
                Previous
              </button>
              <span>
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() =>
                  setCurrentPage((prev) => Math.min(prev + 1, totalPages))
                }
                disabled={currentPage === totalPages}
                className="px-2 py-1 bg-gray-200 rounded-md disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}

      {/* Layer Toggles */}
      <div className="layer-toggles mb-4">
        {Object.keys(layersData)
          .filter(
            (layerName) =>
              layerName !== 'land-use-raster' && layersData[layerName]
          )
          .map((layerName) => (
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
          ))}
      </div>

      {/* Map Container */}
      <MapContainer
        center={defaultPosition}
        zoom={defaultZoom}
        style={{ height: '500px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <LayersControl position="topright">
          {Object.entries(layersData)
            .filter(
              ([layerName, layerData]) =>
                layerName !== 'land-use-raster' &&
                layerData &&
                Array.isArray(layerData.features) &&
                layerData.features.length > 0
            )
            .map(([layerName, layerData]) => (
              <LayersControl.Overlay
                checked={visibleLayers[layerName]}
                name={layerName
                  .split('-')
                  .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(' ')}
                key={layerName}
              >
                <GeoJSON
                  data={layerData}
                  style={layerStyles[layerName]}
                  onEachFeature={(feature, layer) => {
                    if (feature && feature.properties) {
                      layer.bindPopup(
                        `<pre>${JSON.stringify(
                          feature.properties,
                          null,
                          2
                        )}</pre>`
                      );
                    }
                  }}
                />
              </LayersControl.Overlay>
            ))}
        </LayersControl>
        <FeatureHighlighter
          feature={selectedFeature}
          popupContent={popupContent}
        />
      </MapContainer>
    </div>
  );
};

export default MapVisualization;
