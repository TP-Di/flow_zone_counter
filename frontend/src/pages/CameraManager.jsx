import React, { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { cameraAPI } from '../services/api';

function CameraManager() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameraStats, setCameraStats] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    location: '',
    description: '',
  });

  // Load cameras on mount
  useEffect(() => {
    loadCameras();
  }, []);

  // Load stats when camera is selected
  useEffect(() => {
    if (selectedCamera) {
      loadCameraStats(selectedCamera.id);
    }
  }, [selectedCamera]);

  const loadCameras = async () => {
    try {
      const response = await cameraAPI.list();
      setCameras(response.data);
      if (response.data.length > 0 && !selectedCamera) {
        setSelectedCamera(response.data[0]);
      }
    } catch (error) {
      console.error('Error loading cameras:', error);
      alert('Failed to load cameras');
    }
  };

  const loadCameraStats = async (cameraId) => {
    try {
      const response = await cameraAPI.getStats(cameraId);
      setCameraStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const handleCreateCamera = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await cameraAPI.create(formData);
      setShowCreateModal(false);
      setFormData({ name: '', location: '', description: '' });
      await loadCameras();
      alert('Camera created successfully!');
    } catch (error) {
      console.error('Error creating camera:', error);
      alert(error.response?.data?.detail || 'Failed to create camera');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (!selectedCamera) {
      alert('Please select a camera first');
      return;
    }

    setUploadProgress({ current: 0, total: acceptedFiles.length });

    try {
      const response = await cameraAPI.upload(selectedCamera.id, acceptedFiles);
      alert(`Uploaded ${response.data.uploaded} images successfully!`);
      await loadCameraStats(selectedCamera.id);
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Failed to upload images');
    } finally {
      setUploadProgress(null);
    }
  }, [selectedCamera]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: true,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Camera Management</h1>
          <p className="mt-2 text-sm text-gray-600">
            Configure cameras and upload datasets
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition"
        >
          + Create Camera
        </button>
      </div>

      {/* Camera Selector */}
      <div className="bg-white rounded-lg shadow p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Camera
        </label>
        <select
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          value={selectedCamera?.id || ''}
          onChange={(e) => {
            const camera = cameras.find(c => c.id === parseInt(e.target.value));
            setSelectedCamera(camera);
          }}
        >
          <option value="">-- Select a camera --</option>
          {cameras.map((camera) => (
            <option key={camera.id} value={camera.id}>
              {camera.name} {camera.location ? `(${camera.location})` : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Camera Stats */}
      {cameraStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <StatCard
            title="Labeled Images"
            value={cameraStats.labeled_count}
            color="green"
          />
          <StatCard
            title="Unlabeled Images"
            value={cameraStats.unlabeled_count}
            color="yellow"
          />
          <StatCard
            title="Total Images"
            value={cameraStats.total_count}
            color="blue"
          />
        </div>
      )}

      {/* Upload Area */}
      {selectedCamera && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Upload Images</h2>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400'
            }`}
          >
            <input {...getInputProps()} />
            <div className="text-gray-600">
              <svg
                className="mx-auto h-12 w-12 text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 48 48"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 14v20c0 4.418 7.163 8 16 8 1.381 0 2.721-.087 4-.252M8 14c0 4.418 7.163 8 16 8s16-3.582 16-8M8 14c0-4.418 7.163-8 16-8s16 3.582 16 8m0 0v14m0-4c0 4.418-7.163 8-16 8S8 28.418 8 24m32 10v6m0 0v6m0-6h6m-6 0h-6"
                />
              </svg>
              {isDragActive ? (
                <p className="text-lg">Drop images here...</p>
              ) : (
                <div>
                  <p className="text-lg mb-2">
                    Drag & drop images here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports JPG, JPEG, PNG
                  </p>
                </div>
              )}
            </div>
          </div>

          {uploadProgress && (
            <div className="mt-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Uploading...</span>
                <span>{uploadProgress.current} / {uploadProgress.total}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all"
                  style={{
                    width: `${(uploadProgress.current / uploadProgress.total) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Create Camera Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h2 className="text-2xl font-bold mb-4">Create New Camera</h2>
            <form onSubmit={handleCreateCamera} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Camera Name *
                </label>
                <input
                  type="text"
                  required
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Location
                </label>
                <input
                  type="text"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  value={formData.location}
                  onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  rows={3}
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                />
              </div>
              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
                >
                  {loading ? 'Creating...' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ title, value, color }) {
  const colorClasses = {
    green: 'bg-green-100 text-green-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    blue: 'bg-blue-100 text-blue-800',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
        </div>
        <div className={`px-3 py-1 rounded-full ${colorClasses[color]}`}>
          {title.split(' ')[0]}
        </div>
      </div>
    </div>
  );
}

export default CameraManager;
