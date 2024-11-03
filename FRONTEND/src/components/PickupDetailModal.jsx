// components/PickupDetailModal.jsx
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { useState } from 'react';

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

function PickupDetailModal({ pickup, onClose, onUpdate }) {
  const [status, setStatus] = useState(pickup.status);
  const [loading, setLoading] = useState(false);

  const handleStatusChange = async (newStatus) => {
    try {
      setLoading(true);
      await axios.patch(
        `http://localhost:3000/api/specialPickup/${pickup.id}/status`,
        {
          status: newStatus,
        }
      );
      setStatus(newStatus);
      toast.success('Status updated successfully');
      onUpdate();
    } catch (error) {
      toast.error('Failed to update status');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-white rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between items-center">
          <h2 className="text-xl font-bold">Pickup Request Details</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <svg
              className="w-6 h-6"
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
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left Column - Details */}
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Waste Type</h3>
              <p className="mt-1 text-lg">{pickup.wasteType}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500">Quantity</h3>
              <p className="mt-1 text-lg">{pickup.quantity}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500">Description</h3>
              <p className="mt-1 text-gray-700">{pickup.description}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500">Status</h3>
              <select
                value={status}
                onChange={(e) => handleStatusChange(e.target.value)}
                disabled={loading}
                className={`mt-1 px-3 py-1 rounded-full text-sm font-semibold ${
                  STATUS_OPTIONS.find((s) => s.value === status)?.color
                }`}
              >
                {STATUS_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500">Schedule</h3>
              <p className="mt-1 text-lg">
                {format(new Date(pickup.preferredDate), 'PPP')}
              </p>
              <p className="text-gray-600">{pickup.preferredTime}</p>
            </div>
          </div>

          {/* Right Column - Map & Contact */}
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">
                Location
              </h3>
              <div className="h-64 rounded-lg overflow-hidden">
                <MapContainer
                  center={[pickup.location.lat, pickup.location.lng]}
                  zoom={15}
                  className="h-full w-full"
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  />
                  <Marker
                    position={[pickup.location.lat, pickup.location.lng]}
                  />
                </MapContainer>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-500 mb-2">
                Contact Information
              </h3>
              <div className="space-y-2">
                <p className="text-gray-900 font-medium">
                  {pickup.contactName}
                </p>
                <p className="text-gray-600">{pickup.contactPhone}</p>
                <p className="text-gray-600">{pickup.contactEmail}</p>
              </div>
            </div>

            {pickup.photos?.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">
                  Photos
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {pickup.photos.map((photo, index) => (
                    <img
                      key={index}
                      src={`http://localhost:3000/${photo}`}
                      alt={`Pickup photo ${index + 1}`}
                      className="w-full h-32 object-cover rounded-lg"
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default PickupDetailModal;
