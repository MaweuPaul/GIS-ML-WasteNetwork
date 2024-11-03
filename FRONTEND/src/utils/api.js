import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:3000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for handling FormData
api.interceptors.request.use(
  (config) => {
    if (config.data instanceof FormData) {
      config.headers['Content-Type'] = 'multipart/form-data';
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.error || error.message || 'Something went wrong';
    throw new Error(message);
  }
);

const incidentAPI = {
  // Get all incidents with filters
  getIncidents: (params) => {
    return api.get('/incidents', { params });
  },

  // Get single incident
  getIncidentById: (id) => {
    return api.get(`/incidents/${id}`);
  },

  // Create new incident
  createIncident: (data) => {
    const formData = new FormData();

    // Handle regular fields
    Object.keys(data).forEach((key) => {
      if (key !== 'photos') {
        formData.append(
          key,
          typeof data[key] === 'object' ? JSON.stringify(data[key]) : data[key]
        );
      }
    });

    // Handle photos
    if (data.photos) {
      data.photos.forEach((photo) => {
        formData.append('photos', photo);
      });
    }

    return api.post('/incidents', formData);
  },

  // Update incident status
  updateIncidentStatus: (id, status) => {
    return api.patch(`/incidents/${id}/status`, { status });
  },

  // Update incident details
  updateIncident: (id, data) => {
    const formData = new FormData();

    Object.keys(data).forEach((key) => {
      if (key !== 'photos') {
        formData.append(
          key,
          typeof data[key] === 'object' ? JSON.stringify(data[key]) : data[key]
        );
      }
    });

    if (data.photos) {
      data.photos.forEach((photo) => {
        formData.append('photos', photo);
      });
    }

    return api.put(`/incidents/${id}`, formData);
  },

  // Delete incident
  deleteIncident: (id) => {
    return api.delete(`/incidents/${id}`);
  },

  // Get dashboard statistics
  getDashboardStats: () => {
    return api.get('/incidents/stats');
  },

  // Get nearby incidents
  getNearbyIncidents: (lat, lng, distance) => {
    return api.get(`/incidents/nearby/${lat}/${lng}/${distance}`);
  },

  // Export incidents
  exportIncidents: (format = 'csv', filters = {}) => {
    return api.get('/incidents/export', {
      params: { format, ...filters },
      responseType: 'blob',
    });
  },

  // Upload additional photos
  uploadPhotos: (incidentId, photos) => {
    const formData = new FormData();
    photos.forEach((photo) => {
      formData.append('photos', photo);
    });
    return api.post(`/incidents/${incidentId}/photos`, formData);
  },

  // Delete photo
  deletePhoto: (incidentId, photoId) => {
    return api.delete(`/incidents/${incidentId}/photos/${photoId}`);
  },

  // Get incident statistics by type
  getIncidentsByType: () => {
    return api.get('/incidents/by-type');
  },

  // Get incident statistics by priority
  getIncidentsByPriority: () => {
    return api.get('/incidents/by-priority');
  },

  // Get incident statistics by status
  getIncidentsByStatus: () => {
    return api.get('/incidents/by-status');
  },
};

export default incidentAPI;
