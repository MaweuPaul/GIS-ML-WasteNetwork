import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  MapContainer,
  TileLayer,
  Marker,
  useMap,
  useMapEvents,
  Popup,
} from 'react-leaflet';
import { GeoSearchControl, OpenStreetMapProvider } from 'leaflet-geosearch';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-geosearch/dist/geosearch.css';

// Fix for default marker icons
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

// Custom search control styles
const searchControlStyle = `
  .leaflet-control-geosearch {
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    z-index: 1000;
  }
  .leaflet-control-geosearch form {
    background: white;
    padding: 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .leaflet-control-geosearch input {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
  }
`;

function MapWithErrorBoundary({ children }) {
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return (
      <div className="h-64 rounded-lg border border-gray-200 flex items-center justify-center">
        <p className="text-gray-500">Failed to load map</p>
      </div>
    );
  }

  return children;
}

function Incidents() {
  // State declarations
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedIncidents, setSelectedIncidents] = useState([]);
  const [filters, setFilters] = useState({
    type: '',
    priority: '',
    status: '',
  });
  const [pagination, setPagination] = useState({
    currentPage: 1,
    totalPages: 1,
    limit: 10,
  });
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [locationDetails, setLocationDetails] = useState(null);
  const baseUrl = 'http://localhost:3000';

  // Constants
  const incidentTypes = [
    { value: 'ILLEGAL_DUMPING', label: 'Illegal Dumping', icon: '🗑️' },
    { value: 'OVERFLOWING_BIN', label: 'Overflowing Bin', icon: '🚮' },
    { value: 'MISSED_COLLECTION', label: 'Missed Collection', icon: '🚛' },
    { value: 'DAMAGED_BIN', label: 'Damaged Bin', icon: '⚠️' },
    { value: 'HAZARDOUS_WASTE', label: 'Hazardous Waste', icon: '☢️' },
    { value: 'OTHER', label: 'Other Issue', icon: '❓' },
  ];

  const priorityLevels = [
    { value: 'LOW', label: 'Low', color: 'bg-gray-100 text-gray-800' },
    { value: 'MEDIUM', label: 'Medium', color: 'bg-blue-100 text-blue-800' },
    { value: 'HIGH', label: 'High', color: 'bg-orange-100 text-orange-800' },
    { value: 'URGENT', label: 'Urgent', color: 'bg-red-100 text-red-800' },
  ];

  const statusTypes = [
    {
      value: 'PENDING',
      label: 'Pending',
      color: 'bg-yellow-100 text-yellow-800',
    },
    {
      value: 'IN_PROGRESS',
      label: 'In Progress',
      color: 'bg-blue-100 text-blue-800',
    },
    {
      value: 'RESOLVED',
      label: 'Resolved',
      color: 'bg-green-100 text-green-800',
    },
  ];

  // Add custom styles
  useEffect(() => {
    const styleSheet = document.createElement('style');
    styleSheet.textContent = searchControlStyle;
    document.head.appendChild(styleSheet);
    return () => document.head.removeChild(styleSheet);
  }, []);

  // Utility functions
  const getAddressFromCoordinates = async (lat, lng) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
      );
      const data = await response.json();
      return data.display_name;
    } catch (error) {
      console.error('Error getting address:', error);
      return 'Address not found';
    }
  };

  // Location effect
  useEffect(() => {
    if (selectedIncident?.location) {
      const { lat, lng } = selectedIncident.location;
      getAddressFromCoordinates(lat, lng).then((address) => {
        setLocationDetails({
          address,
          coordinates: [lat, lng],
        });
      });
    }
  }, [selectedIncident]);

  useEffect(() => {
    if (selectedIncident?.photos) {
      console.log(
        'Photo URLs:',
        selectedIncident.photos.map((photo) =>
          photo.startsWith('http')
            ? photo
            : `http://localhost:3000/uploads/${photo.split('/').pop()}`
        )
      );
    }
  }, [selectedIncident]);

  // Map Components
  function SearchField() {
    const map = useMap();

    useEffect(() => {
      const provider = new OpenStreetMapProvider();
      const searchControl = new GeoSearchControl({
        provider,
        style: 'bar',
        showMarker: false,
        showPopup: false,
        autoClose: true,
        retainZoomLevel: false,
        animateZoom: true,
        keepResult: false,
        searchLabel: 'Search for location',
      });

      map.addControl(searchControl);

      map.on('geosearch/showlocation', async (e) => {
        const { x, y, label } = e.location;
        setLocationDetails({
          address: label,
          coordinates: [y, x],
        });
      });

      return () => map.removeControl(searchControl);
    }, [map]);

    return null;
  }

  function LocationButton() {
    const map = useMap();

    const handleGetLocation = () => {
      if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(async (position) => {
          const newPosition = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          };
          map.flyTo(newPosition, 16);

          const address = await getAddressFromCoordinates(
            newPosition.lat,
            newPosition.lng
          );

          setLocationDetails({
            address,
            coordinates: [newPosition.lat, newPosition.lng],
          });
        });
      }
    };

    return (
      <button
        onClick={handleGetLocation}
        className="absolute bottom-3 left-5 z-[1000] bg-white p-3 rounded-lg shadow-lg hover:bg-gray-100 transition-colors"
        title="Get my location"
      >
        <svg
          className="w-5 h-5 text-gray-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
          />
        </svg>
      </button>
    );
  }

  // Main functionality
  const fetchIncidents = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${baseUrl}/api/incidents`, {
        params: {
          ...filters,
          page: pagination.currentPage,
          limit: pagination.limit,
        },
      });

      if (response.data && Array.isArray(response.data.incidents)) {
        setIncidents(response.data.incidents);
        setPagination((prev) => ({
          ...prev,
          totalPages: Math.ceil(response.data.total / prev.limit),
        }));
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error fetching incidents:', error);
      toast.error('Failed to fetch incidents');
      setIncidents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIncidents();
  }, [filters, pagination.currentPage]);

  const handleDelete = async (id) => {
    try {
      await axios.delete(`${baseUrl}/api/incidents/${id}`);
      toast.success('Incident deleted successfully');
      fetchIncidents();
      setSelectedIncident(null);
      setIsDeleteModalOpen(false);
    } catch (error) {
      console.error('Error deleting incident:', error);
      toast.error('Failed to delete incident');
    }
  };

  const handleStatusChange = async (id, status) => {
    try {
      await axios.patch(`${baseUrl}/api/incidents/${id}/status`, { status });
      toast.success('Status updated successfully');
      fetchIncidents();
    } catch (error) {
      console.error('Error updating status:', error);
      toast.error('Failed to update status');
    }
  };

  const calculateStats = (data = []) => ({
    total: data.length,
    pending: data.filter((i) => i.status === 'PENDING').length,
    inProgress: data.filter((i) => i.status === 'IN_PROGRESS').length,
    resolved: data.filter((i) => i.status === 'RESOLVED').length,
  });

  const handleExport = () => {
    try {
      const headers = [
        'Type',
        'Description',
        'Priority',
        'Status',
        'Location',
        'Created At',
      ];
      const csvData = incidents.map((incident) => [
        incidentTypes.find((t) => t.value === incident.type)?.label,
        incident.description,
        incident.priority,
        incident.status,
        locationDetails?.address || 'Location not available',
        format(new Date(incident.createdAt), 'MMM d, yyyy HH:mm'),
      ]);

      const csvContent = [
        headers.join(','),
        ...csvData.map((row) => row.join(',')),
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute(
        'download',
        `incidents_${format(new Date(), 'yyyy-MM-dd')}.csv`
      );
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error exporting data:', error);
      toast.error('Failed to export data');
    }
  };

  const stats = calculateStats(incidents);

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Incident Management</h1>
        <button
          onClick={handleExport}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
        >
          Export to CSV
        </button>
      </div>
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <h3 className="text-gray-600 font-medium">Total Incidents</h3>
          <p className="text-4xl font-bold mt-2">{stats.total}</p>
        </div>

        <div className="bg-yellow-50 p-6 rounded-xl shadow-sm">
          <h3 className="text-yellow-600 font-medium">Pending</h3>
          <p className="text-4xl font-bold mt-2">{stats.pending}</p>
        </div>

        <div className="bg-blue-50 p-6 rounded-xl shadow-sm">
          <h3 className="text-blue-600 font-medium">In Progress</h3>
          <p className="text-4xl font-bold mt-2">{stats.inProgress}</p>
        </div>

        <div className="bg-green-50 p-6 rounded-xl shadow-sm">
          <h3 className="text-green-600 font-medium">Resolved</h3>
          <p className="text-4xl font-bold mt-2">{stats.resolved}</p>
        </div>
      </div>
      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <select
          value={filters.type}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, type: e.target.value }))
          }
          className="w-full rounded-lg border-gray-300"
        >
          <option value="">All Types</option>
          {incidentTypes.map((type) => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>

        <select
          value={filters.priority}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, priority: e.target.value }))
          }
          className="w-full rounded-lg border-gray-300"
        >
          <option value="">All Priorities</option>
          {priorityLevels.map((priority) => (
            <option key={priority.value} value={priority.value}>
              {priority.label}
            </option>
          ))}
        </select>

        <select
          value={filters.status}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, status: e.target.value }))
          }
          className="w-full rounded-lg border-gray-300"
        >
          <option value="">All Statuses</option>
          {statusTypes.map((status) => (
            <option key={status.value} value={status.value}>
              {status.label}
            </option>
          ))}
        </select>
      </div>
      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {[
                'Type',
                'Description',
                'Priority',
                'Status',
                'Location',
                'Created At',
                'Actions',
              ].map((header) => (
                <th
                  key={header}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-6 py-4 text-center">
                  <div className="flex justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                  </div>
                </td>
              </tr>
            ) : incidents.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-4 text-center text-gray-500">
                  No incidents found
                </td>
              </tr>
            ) : (
              incidents.map((incident) => (
                <tr key={incident.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-2xl mr-2">
                        {
                          incidentTypes.find((t) => t.value === incident.type)
                            ?.icon
                        }
                      </span>
                      <span>
                        {
                          incidentTypes.find((t) => t.value === incident.type)
                            ?.label
                        }
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">{incident.description}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        priorityLevels.find(
                          (p) => p.value === incident.priority
                        )?.color
                      }`}
                    >
                      {incident.priority}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <select
                      value={incident.status}
                      onChange={(e) =>
                        handleStatusChange(incident.id, e.target.value)
                      }
                      className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        statusTypes.find((s) => s.value === incident.status)
                          ?.color
                      }`}
                    >
                      {statusTypes.map((status) => (
                        <option key={status.value} value={status.value}>
                          {status.label}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <button
                      onClick={() => setSelectedIncident(incident)}
                      className="text-blue-600 hover:text-blue-800 underline"
                    >
                      View on map
                    </button>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {format(new Date(incident.createdAt), 'MMM d, yyyy HH:mm')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <button
                      onClick={() => setSelectedIncident(incident)}
                      className="text-indigo-600 hover:text-indigo-900"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {/* Pagination */}
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-700">
          Showing page {pagination.currentPage} of {pagination.totalPages}
        </span>
        <div className="space-x-2">
          <button
            onClick={() =>
              setPagination((prev) => ({
                ...prev,
                currentPage: prev.currentPage - 1,
              }))
            }
            disabled={pagination.currentPage === 1}
            className="px-4 py-2 border rounded-lg disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() =>
              setPagination((prev) => ({
                ...prev,
                currentPage: prev.currentPage + 1,
              }))
            }
            disabled={pagination.currentPage === pagination.totalPages}
            className="px-4 py-2 border rounded-lg disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>

      {/* Incident Modal */}
      <AnimatePresence>
        {selectedIncident && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white rounded-2xl max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
            >
              {/* Header */}
              <div className="sticky top-0 bg-white px-8 py-6 border-b border-gray-100 flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <span className="text-3xl">
                    {
                      incidentTypes.find(
                        (t) => t.value === selectedIncident.type
                      )?.icon
                    }
                  </span>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">
                      Incident Details
                    </h2>
                    <p className="text-gray-500 text-sm mt-1">
                      {
                        incidentTypes.find(
                          (t) => t.value === selectedIncident.type
                        )?.label
                      }
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedIncident(null)}
                  className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <svg
                    className="w-6 h-6 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {/* Content */}
              <div className="p-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Left Column */}
                  <div className="space-y-6">
                    {/* Description Card */}
                    <div className="bg-gray-50 rounded-xl p-6">
                      <h3 className="text-sm font-medium text-gray-500 mb-3">
                        Description
                      </h3>
                      <p className="text-gray-900">
                        {selectedIncident.description}
                      </p>
                    </div>

                    {/* Status & Priority */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 rounded-xl p-6">
                        <h3 className="text-sm font-medium text-gray-500 mb-3">
                          Priority
                        </h3>
                        <span
                          className={`inline-flex px-4 py-2 rounded-full text-sm font-medium ${
                            priorityLevels.find(
                              (p) => p.value === selectedIncident.priority
                            )?.color
                          }`}
                        >
                          {selectedIncident.priority}
                        </span>
                      </div>
                      <div className="bg-gray-50 rounded-xl p-6">
                        <h3 className="text-sm font-medium text-gray-500 mb-3">
                          Status
                        </h3>
                        <select
                          value={selectedIncident.status}
                          onChange={(e) =>
                            handleStatusChange(
                              selectedIncident.id,
                              e.target.value
                            )
                          }
                          className={`w-full px-4 py-2 rounded-full text-sm font-medium border-2 focus:outline-none ${
                            statusTypes.find(
                              (s) => s.value === selectedIncident.status
                            )?.color
                          }`}
                        >
                          {statusTypes.map((status) => (
                            <option key={status.value} value={status.value}>
                              {status.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Contact Information */}
                    <div className="bg-gray-50 rounded-xl p-6">
                      <h3 className="text-sm font-medium text-gray-500 mb-4">
                        Contact Information
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="bg-white p-2 rounded-full">
                            <svg
                              className="w-5 h-5 text-gray-400"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Name</p>
                            <p className="text-gray-900">
                              {selectedIncident.contactName}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="bg-white p-2 rounded-full">
                            <svg
                              className="w-5 h-5 text-gray-400"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                            </svg>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Phone</p>
                            <p className="text-gray-900">
                              {selectedIncident.contactPhone}
                            </p>
                          </div>
                        </div>
                        {selectedIncident.contactEmail && (
                          <div className="flex items-center gap-3">
                            <div className="bg-white p-2 rounded-full">
                              <svg
                                className="w-5 h-5 text-gray-400"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <div>
                              <p className="text-sm text-gray-500">Email</p>
                              <p className="text-gray-900">
                                {selectedIncident.contactEmail}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Timestamps */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 rounded-xl p-6">
                        <h3 className="text-sm font-medium text-gray-500 mb-2">
                          Created
                        </h3>
                        <p className="text-gray-900">
                          {format(
                            new Date(selectedIncident.createdAt),
                            'MMM d, yyyy HH:mm'
                          )}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-xl p-6">
                        <h3 className="text-sm font-medium text-gray-500 mb-2">
                          Updated
                        </h3>
                        <p className="text-gray-900">
                          {format(
                            new Date(selectedIncident.updatedAt),
                            'MMM d, yyyy HH:mm'
                          )}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Right Column */}
                  <div className="space-y-6">
                    {/* Map */}
                    <div className="bg-gray-50 rounded-xl p-6">
                      <h3 className="text-sm font-medium text-gray-500 mb-3">
                        Location
                      </h3>
                      <p className="text-gray-900 mb-4">
                        {locationDetails?.address}
                      </p>
                      <div className="h-[400px] rounded-xl overflow-hidden shadow-inner">
                        <MapContainer
                          center={
                            locationDetails?.coordinates || [-0.4246, 36.9452]
                          }
                          zoom={15}
                          className="h-full w-full"
                          zoomControl={false}
                        >
                          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                          {locationDetails?.coordinates && (
                            <Marker position={locationDetails.coordinates}>
                              <Popup>{locationDetails.address}</Popup>
                            </Marker>
                          )}
                        </MapContainer>
                      </div>
                      <p className="text-xs text-gray-500 mt-3">
                        Coordinates:{' '}
                        {locationDetails?.coordinates[0].toFixed(6)},{' '}
                        {locationDetails?.coordinates[1].toFixed(6)}
                      </p>
                    </div>

                    {/* Photos */}
                    {selectedIncident.photos?.length > 0 && (
                      <div className="bg-gray-50 rounded-xl p-6">
                        <h3 className="text-sm font-medium text-gray-500 mb-4">
                          Photos
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                          {selectedIncident.photos.map((photo, index) => (
                            <div
                              key={index}
                              className="aspect-square rounded-xl overflow-hidden shadow-sm"
                            >
                              <img
                                src={`http://localhost:3000/${photo}`}
                                alt={`Incident photo ${index + 1}`}
                                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="sticky bottom-0 bg-white px-8 py-6 border-t border-gray-100 flex justify-end gap-4">
                <button
                  onClick={() => setSelectedIncident(null)}
                  className="px-6 py-2.5 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
                <button
                  onClick={() => setIsDeleteModalOpen(true)}
                  className="px-6 py-2.5 text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Delete Incident
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {isDeleteModalOpen && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white rounded-lg max-w-md w-full p-6"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <h2 className="text-xl font-bold mb-4">Confirm Delete</h2>
              <p className="text-gray-600 mb-6">
                Are you sure you want to delete this incident? This action
                cannot be undone.
              </p>
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setIsDeleteModalOpen(false)}
                  className="px-4 py-2 border rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleDelete(selectedIncident.id)}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default Incidents;
