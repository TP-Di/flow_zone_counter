import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';
import { cameraAPI, dashboardAPI, inferenceAPI } from '../services/api';

function Dashboard() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processingImages, setProcessingImages] = useState(false);
  const [availableImages, setAvailableImages] = useState([]);
  const [processProgress, setProcessProgress] = useState({ current: 0, total: 0, stage: '' });

  // Filters
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [logsLimit, setLogsLimit] = useState(50);

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    if (selectedCamera) {
      loadDashboardData();
      loadAvailableImages();
    }
  }, [selectedCamera, startDate, endDate]);

  const loadAvailableImages = async () => {
    if (!selectedCamera) return;

    try {
      // Get labeled images for processing
      const response = await cameraAPI.get(selectedCamera.id);
      // For now, we'll load images from the labeled directory
      // You might want to add an API endpoint to list processed/unprocessed images
      setAvailableImages([]);
    } catch (error) {
      console.error('Error loading images:', error);
    }
  };

  const loadCameras = async () => {
    try {
      const response = await cameraAPI.list();
      setCameras(response.data);
      if (response.data.length > 0) {
        setSelectedCamera(response.data[0]);
      }
    } catch (error) {
      console.error('Error loading cameras:', error);
    }
  };

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Load stats
      const statsResponse = await dashboardAPI.getStats(
        selectedCamera.id,
        startDate || undefined,
        endDate || undefined
      );
      setStats(statsResponse.data);

      // Load logs
      const logsResponse = await dashboardAPI.getLogs({
        camera_id: selectedCamera.id,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        limit: logsLimit,
        offset: 0,
      });
      setLogs(logsResponse.data);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const processImageForDetection = async (imageName) => {
    try {
      await dashboardAPI.processImage(selectedCamera.id, imageName);
      return true;
    } catch (error) {
      console.error(`Error processing ${imageName}:`, error);
      return false;
    }
  };

  const processAllImages = async () => {
    if (!selectedCamera) return;

    if (!window.confirm('Process all images (labeled + unlabeled) to generate comprehensive analytics? This will reset existing logs. This may take a few minutes.')) {
      return;
    }

    setProcessingImages(true);
    setProcessProgress({ current: 0, total: 0, stage: 'Initializing...' });

    try {
      // Reset logs first
      setLogs([]);
      setStats(null);

      // Get camera stats
      const statsResponse = await cameraAPI.getStats(selectedCamera.id);
      const stats = statsResponse.data;

      // Get unlabeled images
      const unlabeledResponse = await inferenceAPI.getUnlabeled(selectedCamera.id);
      const unlabeledImages = unlabeledResponse.data.images || [];

      const totalImages = stats.labeled_count + unlabeledImages.length;

      if (totalImages === 0) {
        alert('No images found. Please upload images first.');
        setProcessingImages(false);
        return;
      }

      setProcessProgress({ current: 0, total: totalImages, stage: 'Processing images...' });

      let processedCount = 0;
      let errorCount = 0;

      // Process ALL labeled images
      for (let i = 0; i < stats.labeled_count; i++) {
        const imageName = `${String(i).padStart(6, '0')}.jpg`;
        const success = await processImageForDetection(imageName);
        if (success) {
          processedCount++;
        } else {
          errorCount++;
        }

        // Update progress
        setProcessProgress({
          current: i + 1,
          total: totalImages,
          stage: `Processing labeled images (${i + 1}/${stats.labeled_count})...`
        });
      }

      // Process ALL unlabeled images
      for (let i = 0; i < unlabeledImages.length; i++) {
        const imageName = unlabeledImages[i].filename;
        const success = await processImageForDetection(imageName);
        if (success) {
          processedCount++;
        } else {
          errorCount++;
        }

        // Update progress
        setProcessProgress({
          current: stats.labeled_count + i + 1,
          total: totalImages,
          stage: `Processing unlabeled images (${i + 1}/${unlabeledImages.length})...`
        });
      }

      setProcessProgress({ current: totalImages, total: totalImages, stage: 'Loading results...' });

      if (processedCount > 0) {
        await loadDashboardData();
        const successMsg = `Successfully processed ALL ${processedCount} images! ${errorCount > 0 ? `\n(${errorCount} failed)` : ''}`;
        alert(successMsg + '\n\nDashboard updated!\n\nNote: Chart shows max 50 aggregated points covering the entire time range.');
      } else {
        alert('No images were processed successfully. Check console for errors.');
      }
    } catch (error) {
      console.error('Error processing images:', error);
      alert('Failed to process images: ' + error.message);
    } finally {
      setProcessingImages(false);
      setProcessProgress({ current: 0, total: 0, stage: '' });
    }
  };

  const exportLogs = async () => {
    try {
      const response = await dashboardAPI.export(
        selectedCamera.id,
        startDate || undefined,
        endDate || undefined
      );

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `detections_${selectedCamera.name}_${new Date().toISOString()}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      alert('Logs exported successfully!');
    } catch (error) {
      console.error('Error exporting logs:', error);
      alert('Failed to export logs');
    }
  };

  // Prepare chart data with aggregation (max 50 points)
  const prepareChartData = () => {
    if (!stats?.time_series || stats.time_series.length === 0) return [];

    const timeSeries = stats.time_series;
    const maxPoints = 50;

    // If we have fewer than maxPoints, show all
    if (timeSeries.length <= maxPoints) {
      return timeSeries.map(item => ({
        time: format(new Date(item.timestamp), 'HH:mm'),
        'Total People': item.total_people,
        'In Zone': item.people_in_zone,
      }));
    }

    // Aggregate data points to fit within maxPoints
    const aggregatedData = [];
    const chunkSize = Math.ceil(timeSeries.length / maxPoints);

    for (let i = 0; i < timeSeries.length; i += chunkSize) {
      const chunk = timeSeries.slice(i, i + chunkSize);

      // Average the values in this chunk
      const avgTotalPeople = Math.round(
        chunk.reduce((sum, item) => sum + item.total_people, 0) / chunk.length
      );
      const avgInZone = Math.round(
        chunk.reduce((sum, item) => sum + item.people_in_zone, 0) / chunk.length
      );

      // Use the first timestamp in the chunk
      aggregatedData.push({
        time: format(new Date(chunk[0].timestamp), 'HH:mm'),
        'Total People': avgTotalPeople,
        'In Zone': avgInZone,
      });
    }

    return aggregatedData;
  };

  const chartData = prepareChartData();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard & Analytics</h1>
        <p className="mt-2 text-sm text-gray-600">
          Real-time people counting and analytics
        </p>
      </div>

      {/* Camera Selector & Filters */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Camera
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              value={selectedCamera?.id || ''}
              onChange={(e) => {
                const camera = cameras.find(c => c.id === parseInt(e.target.value));
                setSelectedCamera(camera);
              }}
            >
              {cameras.map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Start Date
            </label>
            <input
              type="datetime-local"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              End Date
            </label>
            <input
              type="datetime-local"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={loadDashboardData}
              disabled={loading}
              className="w-full px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>

      {/* Process Images Section */}
      {selectedCamera && logs.length === 0 && !processingImages && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-blue-800">No Detection Logs Found</h3>
              <p className="mt-2 text-sm text-blue-700">
                The dashboard shows comprehensive statistics from ALL processed images (labeled + unlabeled).
                <br />Processing generates:
                <br />• People counting with your trained model on ALL images
                <br />• Person tracking with unique IDs
                <br />• Zone crossing analysis across entire dataset
                <br />• Time series chart (automatically aggregated to max 50 points covering full time range)
              </p>
              <div className="mt-4">
                <button
                  onClick={processAllImages}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  🚀 Process All Images (Labeled + Unlabeled)
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Processing Progress */}
      {processingImages && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center mb-4">
            <div className="flex-shrink-0">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
            <div className="ml-4 flex-1">
              <h3 className="text-lg font-semibold text-gray-900">Processing Images</h3>
              <p className="text-sm text-gray-600 mt-1">{processProgress.stage}</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-300 flex items-center justify-end pr-2"
              style={{ width: `${processProgress.total > 0 ? (processProgress.current / processProgress.total) * 100 : 0}%` }}
            >
              <span className="text-xs text-white font-semibold">
                {processProgress.total > 0 ? Math.round((processProgress.current / processProgress.total) * 100) : 0}%
              </span>
            </div>
          </div>

          {/* Progress Text */}
          <p className="text-sm text-gray-600 text-center">
            {processProgress.current} / {processProgress.total} images processed
          </p>
        </div>
      )}

      {/* Quick Process Button for when logs exist */}
      {selectedCamera && logs.length > 0 && (
        <div className="flex justify-end">
          <button
            onClick={processAllImages}
            disabled={processingImages}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {processingImages ? 'Processing...' : '🔄 Reprocess All Images'}
          </button>
        </div>
      )}

      {/* Stats Cards */}
      {stats && logs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <StatCard
            title="Total Unique People"
            value={stats.total_unique_people}
            icon="👥"
            color="blue"
          />
          <StatCard
            title="People in Zone"
            value={stats.current_people_in_zone}
            icon="🎯"
            color="green"
          />
          <StatCard
            title="Crossing Rate"
            value={`${stats.crossing_conversion_rate.toFixed(1)}%`}
            icon="📈"
            color="purple"
          />
        </div>
      )}

      {/* Time Series Chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">People Count Over Time</h2>
            {stats?.time_series && stats.time_series.length > 50 && (
              <span className="text-xs text-gray-500">
                Showing aggregated data ({stats.time_series.length} detections → {chartData.length} points)
              </span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="Total People"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="In Zone"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detection Logs */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Detection Logs</h2>
          <button
            onClick={exportLogs}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            📥 Export CSV
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total People
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  In Zone
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Unique IDs
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {logs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    No detection logs found
                  </td>
                </tr>
              )}

              {logs.map((log) => {
                let uniqueIds = [];
                try {
                  uniqueIds = log.unique_ids ? JSON.parse(log.unique_ids) : [];
                } catch (e) {
                  uniqueIds = [];
                }

                return (
                  <tr key={log.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {format(new Date(log.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {log.image_path.split('/').pop()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {log.total_people}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        {log.people_in_zone}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {uniqueIds.length > 0 ? (
                        <span className="text-xs">
                          {uniqueIds.join(', ')}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {logs.length > 0 && (
          <div className="mt-4 text-sm text-gray-600 text-center">
            Showing {logs.length} most recent detections
          </div>
        )}
      </div>

      {/* No Camera */}
      {!selectedCamera && (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <svg
            className="mx-auto h-12 w-12 text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-gray-600">No cameras configured</p>
          <p className="text-sm text-gray-500 mt-2">Create a camera to start tracking</p>
        </div>
      )}
    </div>
  );
}

function StatCard({ title, value, icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center">
        <div className={`${colorClasses[color]} rounded-full p-3 mr-4`}>
          <span className="text-2xl">{icon}</span>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold mt-1">{value}</p>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
