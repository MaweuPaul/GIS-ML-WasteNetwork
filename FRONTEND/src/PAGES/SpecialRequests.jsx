// pages/SpecialPickups.jsx
import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { format } from 'date-fns';
import { toast } from 'react-hot-toast';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { CSVLink } from 'react-csv';
import { Tooltip } from 'react-tooltip';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import 'leaflet/dist/leaflet.css';
import NewRequestModal from '../components/NewRequestModal';
import PickupDetailModal from '../components/PickupDetailModal';

const API_URL = 'http://localhost:3000/api/specialPickup';

const STATUS_OPTIONS = [
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
    value: 'COMPLETED',
    label: 'Completed',
    color: 'bg-green-100 text-green-800',
  },
  { value: 'CANCELLED', label: 'Cancelled', color: 'bg-red-100 text-red-800' },
];

const WASTE_TYPES = [
  { value: 'GENERAL', label: 'General Waste', icon: '🗑️' },
  { value: 'RECYCLABLE', label: 'Recyclable', icon: '♻️' },
  { value: 'ORGANIC', label: 'Organic Waste', icon: '🌱' },
  { value: 'HAZARDOUS', label: 'Hazardous', icon: '⚠️' },
  { value: 'E_WASTE', label: 'Electronic Waste', icon: '🖥️' },
];

function SpecialPickups() {
  // State Management
  const [pickups, setPickups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPickup, setSelectedPickup] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    status: '',
    type: '',
    date: null,
  });
  const [pagination, setPagination] = useState({
    currentPage: 1,
    itemsPerPage: 10,
    totalItems: 0,
  });
  const [selectedItems, setSelectedItems] = useState([]);
  const [sortConfig, setSortConfig] = useState({
    key: 'createdAt',
    direction: 'desc',
  });
  const [showNewRequestModal, setShowNewRequestModal] = useState(false);

  // Data Fetching
  useEffect(() => {
    fetchPickups();
  }, [pagination.currentPage, filters, sortConfig, searchTerm]);

  const fetchPickups = async () => {
    try {
      setLoading(true);
      const response = await axios.get(API_URL, {
        params: {
          page: pagination.currentPage,
          limit: pagination.itemsPerPage,
          sort: sortConfig.key,
          direction: sortConfig.direction,
          search: searchTerm,
          ...filters,
        },
      });
      setPickups(response.data);
      console.log(response.data);
      setPagination((prev) => ({
        ...prev,
        totalItems: response.data.total,
      }));
    } catch (error) {
      toast.error('Failed to fetch pickup requests');
    } finally {
      setLoading(false);
    }
  };

  // Event Handlers
  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  const handleBulkAction = async (action) => {
    try {
      await axios.post(`${API_URL}/bulk`, {
        ids: selectedItems,
        action,
      });
      toast.success('Bulk action completed successfully');
      fetchPickups();
      setSelectedItems([]);
    } catch (error) {
      toast.error('Failed to perform bulk action');
    }
  };

  // CSV Export Data
  const csvData = pickups.map((pickup) => ({
    ID: pickup.id,
    Type: pickup.type,
    Status: pickup.status,
    Contact: pickup.contactName,
    Phone: pickup.contactPhone,
    Email: pickup.contactEmail,
    Date: format(new Date(pickup.preferredDate), 'PP'),
    Time: pickup.preferredTime,
    Location: `${pickup.location.lat}, ${pickup.location.lng}`,
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 mt-16">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Special Pickup Requests
          </h1>
          <p className="mt-2 text-gray-600">
            Manage and track special waste collection requests
          </p>
        </div>
        <div className="flex gap-4">
          <CSVLink
            data={csvData}
            filename={`special-pickups-${format(new Date(), 'yyyy-MM-dd')}.csv`}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Export CSV
          </CSVLink>
          <button
            onClick={() => setShowNewRequestModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            New Request
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <input
              type="text"
              placeholder="Search requests..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full rounded-lg border-gray-300"
            />
          </div>
          <select
            value={filters.status}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, status: e.target.value }))
            }
            className="w-full rounded-lg border-gray-300"
          >
            <option value="">All Statuses</option>
            {STATUS_OPTIONS.map((status) => (
              <option key={status.value} value={status.value}>
                {status.label}
              </option>
            ))}
          </select>
          <select
            value={filters.type}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, type: e.target.value }))
            }
            className="w-full rounded-lg border-gray-300"
          >
            <option value="">All Waste Types</option>
            {WASTE_TYPES.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
          <DatePicker
            selected={filters.date}
            onChange={(date) => setFilters((prev) => ({ ...prev, date }))}
            className="w-full rounded-lg border-gray-300"
            placeholderText="Select date"
            isClearable
          />
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedItems.length > 0 && (
        <div className="bg-blue-50 rounded-xl p-4 mb-6 flex items-center justify-between">
          <span className="text-blue-700">
            {selectedItems.length} items selected
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => handleBulkAction('cancel')}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Cancel Selected
            </button>
            <button
              onClick={() => handleBulkAction('complete')}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Complete Selected
            </button>
          </div>
        </div>
      )}

      {/* Map View */}
      <div className="relative z-0">
        <h2 className="text-lg font-semibold mb-4">Pickup Locations</h2>
        <div className="h-[400px] rounded-lg overflow-hidden">
          <MapContainer
            center={[-1.2921, 36.8219]}
            zoom={13}
            className="h-full w-full"
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
            {pickups.map((pickup) => (
              <Marker
                key={pickup.id}
                position={[pickup.location.lat, pickup.location.lng]}
              >
                <Popup>
                  <div>
                    <h3 className="font-medium">{pickup.contactName}</h3>
                    <p>{pickup.description}</p>
                    <p className="text-sm text-gray-500">
                      {pickup.preferredTime}
                    </p>
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </div>

      {/* Table View */}
      <div className="bg-white rounded-xl shadow-sm overflow-hidden">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-8 px-6 py-4">
                  <input
                    type="checkbox"
                    checked={selectedItems.length === pickups.length}
                    onChange={(e) => {
                      setSelectedItems(
                        e.target.checked ? pickups.map((p) => p.id) : []
                      );
                    }}
                    className="rounded border-gray-300"
                  />
                </th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Contact
                </th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date & Time
                </th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {pickups.map((pickup) => (
                <tr key={pickup.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <input
                      type="checkbox"
                      checked={selectedItems.includes(pickup.id)}
                      onChange={(e) => {
                        setSelectedItems((prev) =>
                          e.target.checked
                            ? [...prev, pickup.id]
                            : prev.filter((id) => id !== pickup.id)
                        );
                      }}
                      className="rounded border-gray-300"
                    />
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-2xl mr-2">
                        {WASTE_TYPES.find((t) => t.value === pickup.type)?.icon}
                      </span>
                      <span className="font-medium">
                        {
                          WASTE_TYPES.find((t) => t.value === pickup.type)
                            ?.label
                        }
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm">
                      <div className="font-medium text-gray-900">
                        {pickup.contactName}
                      </div>
                      <div className="text-gray-500">{pickup.contactPhone}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm">
                      <div className="font-medium text-gray-900">
                        {format(new Date(pickup.preferredDate), 'PP')}
                      </div>
                      <div className="text-gray-500">
                        {pickup.preferredTime}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-3 py-1 inline-flex text-xs font-semibold rounded-full ${
                        STATUS_OPTIONS.find((s) => s.value === pickup.status)
                          ?.color
                      }`}
                    >
                      {pickup.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <button
                      onClick={() => setSelectedPickup(pickup)}
                      className="text-blue-600 hover:text-blue-900 font-medium"
                    >
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      <div className="mt-6 flex justify-between items-center">
        <div className="text-sm text-gray-700">
          Showing {(pagination.currentPage - 1) * pagination.itemsPerPage + 1}{' '}
          to{' '}
          {Math.min(
            pagination.currentPage * pagination.itemsPerPage,
            pagination.totalItems
          )}{' '}
          of {pagination.totalItems} results
        </div>
        <div className="flex gap-2">
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
            disabled={
              pagination.currentPage * pagination.itemsPerPage >=
              pagination.totalItems
            }
            className="px-4 py-2 border rounded-lg disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>

      {/* Modals */}
      <AnimatePresence>
        {selectedPickup && (
          <PickupDetailModal
            pickup={selectedPickup}
            onClose={() => setSelectedPickup(null)}
            onUpdate={fetchPickups}
          />
        )}
        {showNewRequestModal && (
          <NewRequestModal
            onClose={() => setShowNewRequestModal(false)}
            onSuccess={() => {
              setShowNewRequestModal(false);
              fetchPickups();
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default SpecialPickups;
