import React, { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Line, Circle, Image as KonvaImage } from 'react-konva';
import { cameraAPI, zoneAPI, getImageURL } from '../services/api';

function ZoneDrawer() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [sampleImage, setSampleImage] = useState(null);
  const [imageObj, setImageObj] = useState(null);
  const [points, setPoints] = useState([]);
  const [threshold, setThreshold] = useState(0.3);
  const [isDrawing, setIsDrawing] = useState(false);
  const [existingZone, setExistingZone] = useState(null);
  const [loading, setLoading] = useState(false);

  const stageRef = useRef(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    if (selectedCamera) {
      loadSampleImage();
      loadExistingZone();
    }
  }, [selectedCamera]);

  useEffect(() => {
    if (sampleImage) {
      const img = new window.Image();
      img.crossOrigin = 'anonymous';
      img.src = sampleImage;
      img.onload = () => {
        setImageObj(img);
        // Calculate dimensions to fit container
        const containerWidth = containerRef.current?.offsetWidth || 800;
        const aspectRatio = img.height / img.width;
        const width = Math.min(containerWidth - 40, img.width);
        const height = width * aspectRatio;
        setDimensions({ width, height });
      };
    }
  }, [sampleImage]);

  const loadCameras = async () => {
    try {
      const response = await cameraAPI.list();
      setCameras(response.data);
    } catch (error) {
      console.error('Error loading cameras:', error);
    }
  };

  const loadSampleImage = async (useRandom = false) => {
    try {
      const response = await zoneAPI.getSampleImage(selectedCamera.id, useRandom);
      const imageUrl = getImageURL(response.data.image_path);
      setSampleImage(imageUrl);
    } catch (error) {
      console.error('Error loading sample image:', error);
      alert('No images found for this camera. Please upload images first.');
    }
  };

  const refreshImage = () => {
    loadSampleImage(true);
  };

  const loadExistingZone = async () => {
    try {
      const response = await zoneAPI.get(selectedCamera.id);
      if (response.data) {
        setExistingZone(response.data);
        setPoints(response.data.points);
        setThreshold(response.data.threshold);
      } else {
        setExistingZone(null);
        setPoints([]);
      }
    } catch (error) {
      console.error('Error loading zone:', error);
    }
  };

  const handleCanvasClick = (e) => {
    if (!isDrawing) return;

    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();

    // Scale coordinates to original image dimensions
    const scaleX = imageObj.width / dimensions.width;
    const scaleY = imageObj.height / dimensions.height;

    const scaledX = pos.x * scaleX;
    const scaledY = pos.y * scaleY;

    setPoints([...points, [scaledX, scaledY]]);
  };

  const handleSaveZone = async () => {
    if (points.length < 3) {
      alert('Please draw at least 3 points to form a zone');
      return;
    }

    setLoading(true);
    try {
      await zoneAPI.set({
        camera_id: selectedCamera.id,
        points: points,
        threshold: threshold,
      });
      alert('Zone saved successfully!');
      setIsDrawing(false);
      loadExistingZone();
    } catch (error) {
      console.error('Error saving zone:', error);
      alert('Failed to save zone');
    } finally {
      setLoading(false);
    }
  };

  const handleClearZone = () => {
    setPoints([]);
    setIsDrawing(true);
  };

  const handleDeleteZone = async () => {
    if (!window.confirm('Are you sure you want to delete this zone?')) {
      return;
    }

    try {
      await zoneAPI.delete(selectedCamera.id);
      setPoints([]);
      setExistingZone(null);
      alert('Zone deleted successfully');
    } catch (error) {
      console.error('Error deleting zone:', error);
      alert('Failed to delete zone');
    }
  };

  // Convert points for rendering (scale to canvas size)
  const getCanvasPoints = () => {
    if (!imageObj) return [];

    const scaleX = dimensions.width / imageObj.width;
    const scaleY = dimensions.height / imageObj.height;

    return points.map(([x, y]) => [x * scaleX, y * scaleY]);
  };

  const canvasPoints = getCanvasPoints();
  const flatPoints = canvasPoints.flat();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Zone Configuration</h1>
        <p className="mt-2 text-sm text-gray-600">
          Define counting zones and crossing lines
        </p>
      </div>

      {/* Camera Selector */}
      <div className="bg-white rounded-lg shadow p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Camera
        </label>
        <select
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          value={selectedCamera?.id || ''}
          onChange={(e) => {
            const camera = cameras.find(c => c.id === parseInt(e.target.value));
            setSelectedCamera(camera);
            setPoints([]);
            setIsDrawing(false);
          }}
        >
          <option value="">-- Select a camera --</option>
          {cameras.map((camera) => (
            <option key={camera.id} value={camera.id}>
              {camera.name}
            </option>
          ))}
        </select>
      </div>

      {/* Controls */}
      {selectedCamera && sampleImage && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Threshold (%)
                </label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="5"
                  value={threshold * 100}
                  onChange={(e) => setThreshold(parseFloat(e.target.value) / 100)}
                  className="w-24 px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>
              <div className="text-sm text-gray-600 mt-6">
                Points: {points.length}
              </div>
              <button
                onClick={refreshImage}
                className="mt-6 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 border border-gray-300"
                title="Load a different sample image"
              >
                🔄 Refresh Image
              </button>
            </div>

            <div className="flex space-x-3">
              {!isDrawing && points.length === 0 && (
                <button
                  onClick={() => setIsDrawing(true)}
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                >
                  Start Drawing
                </button>
              )}

              {isDrawing && (
                <>
                  <button
                    onClick={handleClearZone}
                    className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                  >
                    Clear
                  </button>
                  <button
                    onClick={handleSaveZone}
                    disabled={points.length < 3 || loading}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                  >
                    {loading ? 'Saving...' : 'Save Zone'}
                  </button>
                </>
              )}

              {!isDrawing && points.length > 0 && (
                <>
                  <button
                    onClick={() => setIsDrawing(true)}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                  >
                    Edit Zone
                  </button>
                  <button
                    onClick={handleDeleteZone}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                  >
                    Delete Zone
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Instructions */}
          {isDrawing && (
            <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Instructions:</strong> Click on the image to add points. Add at least 3 points to create a polygon zone.
                The zone will automatically close between the first and last point.
              </p>
            </div>
          )}

          {/* Canvas */}
          <div ref={containerRef} className="border border-gray-300 rounded-lg overflow-hidden">
            <Stage
              ref={stageRef}
              width={dimensions.width}
              height={dimensions.height}
              onClick={handleCanvasClick}
              className="cursor-crosshair"
            >
              <Layer>
                {/* Image */}
                {imageObj && (
                  <KonvaImage
                    image={imageObj}
                    width={dimensions.width}
                    height={dimensions.height}
                  />
                )}

                {/* Zone Polygon */}
                {canvasPoints.length > 0 && (
                  <>
                    <Line
                      points={flatPoints}
                      stroke="#3b82f6"
                      strokeWidth={3}
                      closed
                      fill="rgba(59, 130, 246, 0.2)"
                    />
                    {/* Points */}
                    {canvasPoints.map((point, i) => (
                      <Circle
                        key={i}
                        x={point[0]}
                        y={point[1]}
                        radius={6}
                        fill="#3b82f6"
                        stroke="#ffffff"
                        strokeWidth={2}
                      />
                    ))}
                  </>
                )}
              </Layer>
            </Stage>
          </div>
        </div>
      )}

      {/* No Camera Selected */}
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
              d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <p className="text-gray-600">Please select a camera to configure zones</p>
        </div>
      )}
    </div>
  );
}

export default ZoneDrawer;
