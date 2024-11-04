import { useState, useEffect, useRef } from 'react';
import {
  MapContainer,
  TileLayer,
  Polygon,
  ZoomControl,
  FeatureGroup,
  Popup,
  useMap,
  Marker,
} from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import { toast } from 'react-hot-toast';
import axios from 'axios';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';

// Fix for default marker icon issues in Leaflet with React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png',
});

const baseUrl = 'http://localhost:3000';

const DAYS_OF_WEEK = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];

const initialZoneForm = {
  name: '',
  areas: [],
  collectionDays: [],
  time: '',
  coordinates: null,
}; /// Helper function to convert GeoJSON coordinates [lng, lat] to Leaflet [lat, lng]
const convertGeoJSONToLeaflet = (geoJSON) => {
  try {
    // If we receive a GeoJSON object, extract the coordinates
    const coordinates = geoJSON.coordinates?.[0] || geoJSON;

    if (!Array.isArray(coordinates)) {
      console.error('Invalid GeoJSON coordinates:', coordinates);
      return [];
    }

    // Convert the coordinates
    const leafletCoords = coordinates
      .map((coord) => {
        if (Array.isArray(coord) && coord.length === 2) {
          // Swap [longitude, latitude] to [latitude, longitude]
          return [coord[1], coord[0]];
        }
        console.error('Invalid coordinate pair:', coord);
        return null;
      })
      .filter(Boolean);

    return leafletCoords;
  } catch (error) {
    console.error('Error converting coordinates:', error);
    return [];
  }
};

// Helper function to convert Leaflet coordinates [lat, lng] to GeoJSON [lng, lat]
const convertLeafletToGeoJSON = (leafletCoords) => {
  try {
    if (!Array.isArray(leafletCoords)) {
      console.error('Invalid Leaflet coordinates:', leafletCoords);
      return [];
    }

    return leafletCoords
      .map((coord) => {
        if (Array.isArray(coord) && coord.length === 2) {
          // Swap [latitude, longitude] to [longitude, latitude]
          return [coord[1], coord[0]];
        }
        console.error('Invalid coordinate pair:', coord);
        return null;
      })
      .filter(Boolean);
  } catch (error) {
    console.error('Error converting coordinates:', error);
    return [];
  }
};

// Map Editor Component for handling edit mode
const MapEditor = ({ zone, onUpdate }) => {
  const map = useMap();
  const featureGroupRef = useRef(null);

  useEffect(() => {
    if (
      zone &&
      zone.coordinates &&
      zone.coordinates.type === 'Polygon' &&
      Array.isArray(zone.coordinates.coordinates)
    ) {
      const leafletCoords = convertGeoJSONToLeaflet(
        zone.coordinates.coordinates[0]
      );
      if (leafletCoords.length > 0) {
        const layer = L.polygon(leafletCoords);
        featureGroupRef.current.clearLayers();
        featureGroupRef.current.addLayer(layer);
        map.fitBounds(layer.getBounds(), { padding: [50, 50] });
      } else {
        toast.error('Invalid zone coordinates. Cannot edit this zone.');
      }
    }
  }, [zone, map]);

  const handleEdit = (e) => {
    const layers = e.layers;
    layers.eachLayer((layer) => {
      const coordinates = layer
        .getLatLngs()[0]
        .map((latLng) => [latLng.lat, latLng.lng]);
      onUpdate(coordinates);
    });
  };

  return (
    <FeatureGroup ref={featureGroupRef}>
      <EditControl
        position="topright"
        onEdited={handleEdit}
        edit={{
          featureGroup: featureGroupRef.current,
          edit: true,
          remove: false,
        }}
        draw={{
          rectangle: false,
          circle: false,
          circlemarker: false,
          marker: false,
          polyline: false,
          polygon: false,
        }}
      />
    </FeatureGroup>
  );
};

function CollectionScheduleManager() {
  const [zones, setZones] = useState([]);
  const [selectedZone, setSelectedZone] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [zoneForm, setZoneForm] = useState(initialZoneForm);
  const [drawnLayer, setDrawnLayer] = useState(null);
  const mapRef = useRef(null);
  const featureGroupRef = useRef(null);

  useEffect(() => {
    fetchZones();
  }, []);

  const fetchZones = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${baseUrl}/api/zones`);
      setZones(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      console.error('Fetch error:', error);
      toast.error('Failed to fetch zones');
    } finally {
      setLoading(false);
    }
  };

  const fitMapToBounds = (coordinates) => {
    if (
      mapRef.current &&
      Array.isArray(coordinates) &&
      coordinates.length > 0
    ) {
      const bounds = L.latLngBounds(coordinates);
      mapRef.current.fitBounds(bounds, {
        padding: [50, 50],
        maxZoom: 15,
        duration: 0.5,
      });
    } else {
      console.error('Invalid coordinates for fitBounds:', coordinates);
      toast.error('Cannot fit map to invalid bounds.');
    }
  };
  const handleDrawCreate = (e) => {
    const layer = e.layer;
    setDrawnLayer(layer);
    const geoJSON = layer.toGeoJSON();
    if (
      geoJSON &&
      geoJSON.geometry &&
      geoJSON.geometry.coordinates &&
      Array.isArray(geoJSON.geometry.coordinates[0])
    ) {
      const leafletCoords = convertGeoJSONToLeaflet(
        geoJSON.geometry.coordinates[0]
      );
      setZoneForm((prev) => ({ ...prev, coordinates: leafletCoords }));
      fitMapToBounds(leafletCoords);
      toast.success(
        'Zone area drawn successfully. Please fill in the details.'
      );
    } else {
      toast.error('Invalid geometry drawn. Please try again.');
    }
  };

  const handlePolygonUpdate = (coordinates) => {
    setZoneForm((prev) => ({ ...prev, coordinates }));
    toast.success('Zone area updated. Save changes to confirm.');
  };

  const handleZoneSubmit = async (e) => {
    e.preventDefault();
    try {
      if (!zoneForm.coordinates || zoneForm.coordinates.length === 0) {
        toast.error('Zone coordinates are required.');
        return;
      }

      // Convert Leaflet [lat, lng] to GeoJSON [lng, lat]
      const geoJSONCoordinates = [
        convertLeafletToGeoJSON(zoneForm.coordinates),
      ];

      const payload = {
        name: zoneForm.name,
        areas: Array.isArray(zoneForm.areas)
          ? zoneForm.areas
          : [zoneForm.areas],
        collectionDays: zoneForm.collectionDays,
        time: zoneForm.time,
        coordinates: {
          type: 'Polygon',
          coordinates: geoJSONCoordinates,
        },
      };

      if (editMode && selectedZone) {
        await axios.put(`${baseUrl}/api/zones/${selectedZone.id}`, payload);
        toast.success('Zone updated successfully');
      } else {
        await axios.post(`${baseUrl}/api/zones`, payload);
        toast.success('Zone created successfully');
      }
      fetchZones();
      resetForm();
    } catch (error) {
      console.error('Submit error:', error);
      toast.error(editMode ? 'Failed to update zone' : 'Failed to create zone');
    }
  };

  const handleDeleteZone = async (zoneId) => {
    if (window.confirm('Are you sure you want to delete this zone?')) {
      try {
        await axios.delete(`${baseUrl}/api/zones/${zoneId}`);
        toast.success('Zone deleted successfully');
        fetchZones();
        if (selectedZone?.id === zoneId) {
          resetForm();
        }
      } catch (error) {
        console.error('Delete error:', error);
        toast.error('Failed to delete zone');
      }
    }
  };
  const handleZoneClick = (zone) => {
    try {
      setSelectedZone(zone);

      if (
        zone?.coordinates?.type === 'Polygon' &&
        Array.isArray(zone?.coordinates?.coordinates) &&
        zone.coordinates.coordinates[0]
      ) {
        const leafletCoords = convertGeoJSONToLeaflet(
          zone.coordinates.coordinates[0]
        );

        if (leafletCoords && leafletCoords.length > 0) {
          setZoneForm({
            name: zone.name,
            areas: Array.isArray(zone.areas) ? zone.areas : [zone.areas],
            collectionDays: zone.collectionDays,
            time: zone.time,
            coordinates: leafletCoords,
          });
          setEditMode(true);
          setIsDrawing(false);
          fitMapToBounds(leafletCoords);
        } else {
          toast.error('Invalid zone coordinates. Cannot select this zone.');
        }
      } else {
        toast.error('Invalid zone geometry.');
      }
    } catch (error) {
      console.error('Error handling zone click:', error);
      toast.error('Error selecting zone');
    }
  };

  const resetForm = () => {
    setZoneForm(initialZoneForm);
    setEditMode(false);
    setSelectedZone(null);
    setIsDrawing(false);
    if (drawnLayer) {
      drawnLayer.remove();
      setDrawnLayer(null);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Collection Schedule Manager
          </h1>
          <p className="mt-2 text-gray-600">
            Manage waste collection zones and schedules
          </p>
        </div>
        {!editMode && (
          <button
            onClick={() => setIsDrawing(!isDrawing)}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 ${
              isDrawing
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isDrawing ? (
              <>
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
                Cancel Drawing
              </>
            ) : (
              <>
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                Add New Zone
              </>
            )}
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="space-y-6">
          {/* Zones List */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Zones</h2>
            <div className="space-y-2">
              {zones.length > 0 ? (
                zones.map((zone) => (
                  <div
                    key={zone.id}
                    className={`p-4 rounded-lg transition-all cursor-pointer ${
                      selectedZone?.id === zone.id
                        ? 'bg-blue-50 border-2 border-blue-500'
                        : 'bg-gray-50 hover:bg-gray-100'
                    }`}
                    onClick={() => handleZoneClick(zone)}
                  >
                    <div className="flex justify-between items-center">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900">
                          {zone.name}
                        </h3>
                        <p className="text-sm text-gray-500 mt-1">
                          Areas: {zone.areas.join(', ')}
                        </p>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-gray-500">
                              Collection Days:
                            </span>
                            <p className="font-medium">
                              {zone.collectionDays.join(', ')}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Time:</span>
                            <p className="font-medium">{zone.time}</p>
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col gap-2 ml-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteZone(zone.id);
                          }}
                          className="p-2 text-gray-400 hover:text-red-600 rounded-full hover:bg-red-50"
                          title="Delete Zone"
                        >
                          <svg
                            className="w-5 h-5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 py-4">
                  No zones found. Create one by clicking "Add New Zone"
                </div>
              )}
            </div>
          </div>

          {/* Zone Form */}
          {(isDrawing || editMode) && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">
                  {editMode ? 'Edit Zone' : 'Add New Zone'}
                </h2>
                {editMode && (
                  <button
                    onClick={resetForm}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                )}
              </div>{' '}
              <form onSubmit={handleZoneSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Zone Name
                  </label>
                  <input
                    type="text"
                    value={zoneForm.name}
                    onChange={(e) =>
                      setZoneForm({ ...zoneForm, name: e.target.value })
                    }
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Areas
                  </label>
                  <input
                    type="text"
                    value={
                      Array.isArray(zoneForm.areas)
                        ? zoneForm.areas[0]
                        : zoneForm.areas
                    }
                    onChange={(e) =>
                      setZoneForm({ ...zoneForm, areas: [e.target.value] })
                    }
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Enter area name"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Collection Days
                  </label>
                  <select
                    multiple
                    value={zoneForm.collectionDays}
                    onChange={(e) =>
                      setZoneForm({
                        ...zoneForm,
                        collectionDays: Array.from(
                          e.target.selectedOptions,
                          (option) => option.value
                        ),
                      })
                    }
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  >
                    {DAYS_OF_WEEK.map((day) => (
                      <option key={day} value={day}>
                        {day}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-sm text-gray-500">
                    Hold Ctrl/Cmd to select multiple days
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Collection Time
                  </label>
                  <input
                    type="text"
                    value={zoneForm.time}
                    onChange={(e) =>
                      setZoneForm({ ...zoneForm, time: e.target.value })
                    }
                    placeholder="e.g., 6:00 AM - 10:00 AM"
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>

                <div className="flex justify-end gap-2 mt-6">
                  <button
                    type="button"
                    onClick={resetForm}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    disabled={
                      !zoneForm.coordinates || zoneForm.coordinates.length === 0
                    }
                  >
                    {editMode ? 'Update Zone' : 'Create Zone'}
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>

        {/* Map Section */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className="p-6">
              <h2 className="text-lg font-semibold mb-4">Zone Map</h2>
              <div className="h-[600px] rounded-lg overflow-hidden">
                <MapContainer
                  center={[-0.4169, 36.9514]}
                  zoom={13}
                  className="h-full w-full"
                  ref={mapRef}
                  scrollWheelZoom={true}
                  zoomControl={false}
                  minZoom={5}
                  maxZoom={18}
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  />

                  {zones.map((zone) => {
                    if (
                      zone?.coordinates?.type === 'Polygon' &&
                      Array.isArray(zone?.coordinates?.coordinates) &&
                      zone.coordinates.coordinates[0]
                    ) {
                      try {
                        const leafletCoords = convertGeoJSONToLeaflet(
                          zone.coordinates.coordinates[0]
                        );

                        if (!leafletCoords || leafletCoords.length === 0) {
                          console.error(
                            'Invalid coordinates for zone:',
                            zone.id
                          );
                          return null;
                        }

                        return (
                          <Polygon
                            key={zone.id}
                            positions={leafletCoords}
                            pathOptions={{
                              color:
                                selectedZone?.id === zone.id
                                  ? '#2563EB'
                                  : '#6B7280',
                              fillColor:
                                selectedZone?.id === zone.id
                                  ? '#BFDBFE'
                                  : '#F3F4F6',
                              fillOpacity:
                                selectedZone?.id === zone.id ? 0.4 : 0.2,
                              weight: selectedZone?.id === zone.id ? 3 : 1,
                            }}
                            eventHandlers={{
                              click: () => handleZoneClick(zone),
                            }}
                          >
                            <Popup>
                              <div className="p-2">
                                <h3 className="font-medium">{zone.name}</h3>
                                <p className="text-sm text-gray-500">
                                  Areas: {zone.areas.join(', ')}
                                </p>
                                <p className="text-sm">
                                  Collection: {zone.collectionDays.join(', ')}{' '}
                                  at {zone.time}
                                </p>
                              </div>
                            </Popup>
                          </Polygon>
                        );
                      } catch (error) {
                        console.error('Error rendering zone:', zone.id, error);
                        return null;
                      }
                    }
                    return null;
                  })}
                  {editMode ? (
                    selectedZone && selectedZone.coordinates ? (
                      <MapEditor
                        zone={selectedZone}
                        onUpdate={handlePolygonUpdate}
                      />
                    ) : null
                  ) : (
                    isDrawing && (
                      <FeatureGroup ref={featureGroupRef}>
                        <EditControl
                          position="topright"
                          onCreated={handleDrawCreate}
                          draw={{
                            rectangle: false,
                            circle: false,
                            circlemarker: false,
                            marker: false,
                            polyline: false,
                            polygon: {
                              allowIntersection: false,
                              drawError: {
                                color: '#e1e4e8',
                                message:
                                  "<strong>Oh snap!</strong> you can't draw that!",
                              },
                              shapeOptions: {
                                color: '#2563EB',
                                fillOpacity: 0.3,
                              },
                            },
                          }}
                          edit={false}
                        />
                      </FeatureGroup>
                    )
                  )}
                </MapContainer>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CollectionScheduleManager;
