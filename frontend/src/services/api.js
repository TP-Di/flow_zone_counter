/**
 * API service for communication with backend
 */
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Camera endpoints
export const cameraAPI = {
  create: (data) => api.post('/api/cameras/create', data),
  list: () => api.get('/api/cameras/list'),
  get: (id) => api.get(`/api/cameras/${id}`),
  getStats: (id) => api.get(`/api/cameras/${id}/stats`),
  upload: (id, files) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    return api.post(`/api/cameras/${id}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  delete: (id) => api.delete(`/api/cameras/${id}`),
};

// Zone endpoints
export const zoneAPI = {
  set: (data) => api.post('/api/zones/set', data),
  get: (cameraId) => api.get(`/api/zones/${cameraId}`),
  delete: (cameraId) => api.delete(`/api/zones/${cameraId}`),
  getSampleImage: (cameraId, useRandom = false) =>
    api.get(`/api/zones/${cameraId}/sample-image`, { params: { use_random: useRandom } }),
};

// Inference endpoints
export const inferenceAPI = {
  run: (data) => api.post('/api/inference/run', data),
  approve: (data) => api.post('/api/inference/approve', data),
  getUnlabeled: (cameraId) => api.get(`/api/inference/unlabeled/${cameraId}`),
};

// Training endpoints
export const trainingAPI = {
  start: (data) => api.post('/api/training/start', data),
  getStatus: (jobId) => api.get(`/api/training/status/${jobId}`),
  getCameraJobs: (cameraId) => api.get(`/api/training/camera/${cameraId}/jobs`),
};

// Dashboard endpoints
export const dashboardAPI = {
  getStats: (cameraId, startDate, endDate) => {
    const params = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    return api.get(`/api/dashboard/stats/${cameraId}`, { params });
  },
  getLogs: (filters) => api.post('/api/dashboard/logs', filters),
  export: (cameraId, startDate, endDate) => {
    const params = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    return api.get(`/api/dashboard/export/${cameraId}`, {
      params,
      responseType: 'blob',
    });
  },
  processImage: (cameraId, imageName) =>
    api.post(`/api/dashboard/process-image/${cameraId}?image_name=${imageName}`),
};

// Utility function to get image URL
export const getImageURL = (imagePath) => {
  // Add 'cameras/' prefix if not already present
  const path = imagePath.startsWith('cameras/') ? imagePath : `cameras/${imagePath}`;
  return `${API_BASE_URL}/${path}`;
};

export default api;
