// components/NewRequestModal.jsx
import { useState } from 'react';
import { motion } from 'framer-motion';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import DatePicker from 'react-datepicker';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import axios from 'axios';
import { toast } from 'react-hot-toast';

const validationSchema = Yup.object({
  wasteType: Yup.string().required('Required'),
  quantity: Yup.number().required('Required').min(1, 'Must be at least 1'),
  description: Yup.string()
    .required('Required')
    .min(10, 'Must be at least 10 characters'),
  preferredDate: Yup.date()
    .required('Required')
    .min(new Date(), 'Cannot be in the past'),
  preferredTime: Yup.string().required('Required'),
  location: Yup.object().required('Please select a location'),
  contactName: Yup.string().required('Required'),
  contactPhone: Yup.string().required('Required'),
  contactEmail: Yup.string().email('Invalid email').required('Required'),
});

function LocationMarker({ onLocationSelect }) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng);
    },
  });
  return null;
}

function NewRequestModal({ onClose, onSuccess }) {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);

  const formik = useFormik({
    initialValues: {
      wasteType: '',
      quantity: '',
      description: '',
      preferredDate: null,
      preferredTime: '',
      location: null,
      contactName: '',
      contactPhone: '',
      contactEmail: '',
    },
    validationSchema,
    onSubmit: async (values) => {
      try {
        const formData = new FormData();

        // Append form fields
        Object.keys(values).forEach((key) => {
          if (key === 'location') {
            formData.append(key, JSON.stringify(values[key]));
          } else if (key === 'preferredDate') {
            formData.append(key, values[key].toISOString());
          } else {
            formData.append(key, values[key]);
          }
        });

        // Append photos
        selectedFiles.forEach((file) => {
          formData.append('photos', file);
        });

        await axios.post('http://localhost:3000/api/specialPickup', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        toast.success('Pickup request created successfully');
        onSuccess();
      } catch (error) {
        toast.error('Failed to create pickup request');
      }
    },
  });

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
        <form onSubmit={formik.handleSubmit}>
          {/* Form content */}
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Column */}
            <div className="space-y-6">
              {/* Waste Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Waste Type
                </label>
                <select
                  {...formik.getFieldProps('wasteType')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
                >
                  <option value="">Select a type</option>
                  {/* Add waste type options */}
                </select>
                {formik.touched.wasteType && formik.errors.wasteType && (
                  <div className="mt-1 text-sm text-red-600">
                    {formik.errors.wasteType}
                  </div>
                )}
              </div>

              {/* Other form fields... */}
            </div>

            {/* Right Column */}
            <div className="space-y-6">
              {/* Map */}
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Location
                </label>
                <div className="mt-1 h-64 rounded-lg overflow-hidden">
                  <MapContainer
                    center={[-1.2921, 36.8219]}
                    zoom={13}
                    className="h-full w-full"
                  >
                    <TileLayer
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    />
                    <LocationMarker
                      onLocationSelect={(latlng) => {
                        setSelectedLocation(latlng);
                        formik.setFieldValue('location', latlng);
                      }}
                    />
                    {selectedLocation && <Marker position={selectedLocation} />}
                  </MapContainer>
                </div>
              </div>

              {/* Photos */}
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Photos (Optional)
                </label>
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={(e) => setSelectedFiles(Array.from(e.target.files))}
                  className="mt-1 block w-full"
                />
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 bg-gray-50 border-t flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              disabled={formik.isSubmitting}
            >
              {formik.isSubmitting ? 'Creating...' : 'Create Request'}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
}

export default NewRequestModal;
