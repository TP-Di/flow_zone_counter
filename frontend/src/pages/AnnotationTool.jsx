import React, { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage } from 'react-konva';
import { cameraAPI, inferenceAPI, trainingAPI, getImageURL } from '../services/api';

function AnnotationTool() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [unlabeledImages, setUnlabeledImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [imageObj, setImageObj] = useState(null);
  const [bboxes, setBboxes] = useState([]);
  const [selectedBboxIndex, setSelectedBboxIndex] = useState(null);
  const [isRunningInference, setIsRunningInference] = useState(false);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [newBbox, setNewBbox] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const stageRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    if (selectedCamera) {
      loadUnlabeledImages();
    }
  }, [selectedCamera]);

  useEffect(() => {
    if (unlabeledImages.length > 0 && currentImageIndex < unlabeledImages.length) {
      loadCurrentImage();
      // Reset drawing mode and clear bboxes when navigating images
      setIsDrawingMode(false);
      setIsDrawing(false);
      setNewBbox(null);
    }
  }, [currentImageIndex, unlabeledImages]);

  const loadCameras = async () => {
    try {
      const response = await cameraAPI.list();
      setCameras(response.data);
    } catch (error) {
      console.error('Error loading cameras:', error);
    }
  };

  const loadUnlabeledImages = async () => {
    try {
      const response = await inferenceAPI.getUnlabeled(selectedCamera.id);
      setUnlabeledImages(response.data.images);
      setCurrentImageIndex(0);
      setBboxes([]);
    } catch (error) {
      console.error('Error loading unlabeled images:', error);
    }
  };

  const loadCurrentImage = () => {
    if (unlabeledImages.length === 0) return;

    const currentImage = unlabeledImages[currentImageIndex];
    const imageUrl = getImageURL(currentImage.path);

    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    img.onload = () => {
      setImageObj(img);
      const containerWidth = containerRef.current?.offsetWidth || 800;
      const aspectRatio = img.height / img.width;
      const width = Math.min(containerWidth - 40, img.width);
      const height = width * aspectRatio;
      setDimensions({ width, height });

      // Load inference results if available
      if (inferenceResults) {
        const result = inferenceResults.results.find(
          r => r.image_name === currentImage.filename
        );
        if (result) {
          const scaledBboxes = result.detections.map((det, idx) => ({
            id: idx,
            x1: det.points[0][0],
            y1: det.points[0][1],
            x2: det.points[1][0],
            y2: det.points[1][1],
            approved: true,
            confidence: det.confidence,
          }));
          setBboxes(scaledBboxes);
        } else {
          setBboxes([]);
        }
      }
    };
  };

  const runInference = async () => {
    setIsRunningInference(true);
    // Reset drawing mode and clear any drawn bboxes
    setIsDrawingMode(false);
    setIsDrawing(false);
    setNewBbox(null);
    setBboxes([]);
    try {
      const response = await inferenceAPI.run({
        camera_id: selectedCamera.id,
        confidence: 0.25,
      });
      setInferenceResults(response.data);
      loadCurrentImage();
      alert(`Inference completed on ${response.data.total_images} images!`);
    } catch (error) {
      console.error('Error running inference:', error);
      alert('Failed to run inference');
    } finally {
      setIsRunningInference(false);
    }
  };

  const approveAnnotation = async () => {
    if (bboxes.length === 0) {
      if (!window.confirm('No detections found. Save empty annotation?')) {
        return;
      }
    }

    const currentImage = unlabeledImages[currentImageIndex];
    const img = imageObj;

    // Prepare annotation data
    const annotation = {
      camera_id: selectedCamera.id,
      image_name: currentImage.filename,
      annotations: {
        shapes: bboxes
          .filter(b => b.approved)
          .map(b => ({
            label: 'person',
            points: [[b.x1, b.y1], [b.x2, b.y2]],
            shape_type: 'rectangle',
          })),
        imagePath: currentImage.filename,
        imageHeight: img.height,
        imageWidth: img.width,
      },
    };

    try {
      await inferenceAPI.approve(annotation);

      // Remove from unlabeled list
      const newUnlabeled = unlabeledImages.filter((_, idx) => idx !== currentImageIndex);
      setUnlabeledImages(newUnlabeled);

      // Clear bboxes and reset drawing mode
      setBboxes([]);
      setIsDrawingMode(false);
      setIsDrawing(false);
      setNewBbox(null);

      if (newUnlabeled.length === 0) {
        alert('All images processed!');
        setImageObj(null);
      } else {
        // Force reload of current image (which is now the next one)
        if (currentImageIndex >= newUnlabeled.length) {
          setCurrentImageIndex(newUnlabeled.length - 1);
        } else {
          // Trigger reload by updating index
          setCurrentImageIndex(currentImageIndex);
        }
      }
    } catch (error) {
      console.error('Error approving annotation:', error);
      alert('Failed to save annotation');
    }
  };

  const toggleBboxApproval = (index) => {
    const newBboxes = [...bboxes];
    newBboxes[index].approved = !newBboxes[index].approved;
    setBboxes(newBboxes);
  };

  const deleteBbox = (index) => {
    setBboxes(bboxes.filter((_, idx) => idx !== index));
  };

  const startTraining = async () => {
    if (!window.confirm('Start model training? This may take several minutes.')) {
      return;
    }

    try {
      const response = await trainingAPI.start({
        camera_id: selectedCamera.id,
        epochs: 50,
        batch_size: 16,
        image_size: 640,
      });
      setTrainingStatus(response.data);
      alert('Training started! Job ID: ' + response.data.job_id);

      // Poll for status
      pollTrainingStatus(response.data.job_id);
    } catch (error) {
      console.error('Error starting training:', error);
      alert(error.response?.data?.detail || 'Failed to start training');
    }
  };

  const pollTrainingStatus = async (jobId) => {
    const interval = setInterval(async () => {
      try {
        const response = await trainingAPI.getStatus(jobId);
        setTrainingStatus(response.data);

        if (response.data.status === 'completed' || response.data.status === 'failed') {
          clearInterval(interval);
          alert(`Training ${response.data.status}!`);
        }
      } catch (error) {
        console.error('Error polling status:', error);
        clearInterval(interval);
      }
    }, 5000); // Poll every 5 seconds
  };

  const handleMouseDown = (e) => {
    if (!isDrawingMode || !imageObj) return;

    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();

    // Convert canvas coordinates to image coordinates
    const scaleX = imageObj.width / dimensions.width;
    const scaleY = imageObj.height / dimensions.height;
    const imgX = pos.x * scaleX;
    const imgY = pos.y * scaleY;

    setIsDrawing(true);
    setNewBbox({ x1: imgX, y1: imgY, x2: imgX, y2: imgY });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !imageObj || !newBbox) return;

    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();

    // Convert canvas coordinates to image coordinates
    const scaleX = imageObj.width / dimensions.width;
    const scaleY = imageObj.height / dimensions.height;
    const imgX = pos.x * scaleX;
    const imgY = pos.y * scaleY;

    setNewBbox({ ...newBbox, x2: imgX, y2: imgY });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !newBbox || !imageObj) return;

    // Ensure minimum size
    const minSize = 10;
    const width = Math.abs(newBbox.x2 - newBbox.x1);
    const height = Math.abs(newBbox.y2 - newBbox.y1);

    if (width > minSize && height > minSize) {
      // Normalize coordinates (ensure x1 < x2 and y1 < y2)
      const normalizedBbox = {
        id: bboxes.length,
        x1: Math.min(newBbox.x1, newBbox.x2),
        y1: Math.min(newBbox.y1, newBbox.y2),
        x2: Math.max(newBbox.x1, newBbox.x2),
        y2: Math.max(newBbox.y1, newBbox.y2),
        approved: true,
        confidence: null,
      };

      setBboxes([...bboxes, normalizedBbox]);
    }

    setIsDrawing(false);
    setNewBbox(null);
  };

  const addManualBbox = () => {
    setIsDrawingMode(true);
  };

  const cancelDrawing = () => {
    setIsDrawingMode(false);
    setIsDrawing(false);
    setNewBbox(null);
  };

  const getCanvasBboxes = () => {
    if (!imageObj) return [];

    const scaleX = dimensions.width / imageObj.width;
    const scaleY = dimensions.height / imageObj.height;

    const existingBboxes = bboxes.map(bbox => ({
      ...bbox,
      x: bbox.x1 * scaleX,
      y: bbox.y1 * scaleY,
      width: (bbox.x2 - bbox.x1) * scaleX,
      height: (bbox.y2 - bbox.y1) * scaleY,
    }));

    // Add the bbox being drawn
    if (newBbox && isDrawing) {
      const x1 = Math.min(newBbox.x1, newBbox.x2) * scaleX;
      const y1 = Math.min(newBbox.y1, newBbox.y2) * scaleY;
      const x2 = Math.max(newBbox.x1, newBbox.x2) * scaleX;
      const y2 = Math.max(newBbox.y1, newBbox.y2) * scaleY;

      existingBboxes.push({
        id: 'drawing',
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
        approved: true,
        isDrawing: true,
      });
    }

    return existingBboxes;
  };

  const canvasBboxes = getCanvasBboxes();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Annotation Tool</h1>
          <p className="mt-2 text-sm text-gray-600">
            Review and correct model predictions
          </p>
        </div>
        <div className="flex space-x-3">
          {selectedCamera && unlabeledImages.length > 0 && imageObj && (
            <>
              <button
                onClick={isDrawingMode ? cancelDrawing : addManualBbox}
                className={`px-4 py-2 rounded-lg ${
                  isDrawingMode
                    ? 'bg-gray-600 text-white hover:bg-gray-700'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {isDrawingMode ? 'Cancel Drawing' : 'Draw Bbox'}
              </button>
              <button
                onClick={runInference}
                disabled={isRunningInference || isDrawingMode}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
              >
                {isRunningInference ? 'Running...' : 'Start Analysis'}
              </button>
            </>
          )}
          {selectedCamera && (
            <button
              onClick={startTraining}
              disabled={trainingStatus?.status === 'running'}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {trainingStatus?.status === 'running' ? 'Training...' : 'Train Model'}
            </button>
          )}
        </div>
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
            setInferenceResults(null);
            setBboxes([]);
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

      {/* Training Status */}
      {trainingStatus && trainingStatus.status === 'running' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
            <div>
              <p className="font-medium text-blue-900">Training in progress...</p>
              <p className="text-sm text-blue-700">Job ID: {trainingStatus.job_id}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      {selectedCamera && unlabeledImages.length > 0 && imageObj && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Image Canvas */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">
                Image {currentImageIndex + 1} / {unlabeledImages.length}
              </h2>
              <div className="flex space-x-2">
                <button
                  onClick={() => setCurrentImageIndex(Math.max(0, currentImageIndex - 1))}
                  disabled={currentImageIndex === 0}
                  className="px-3 py-1 border border-gray-300 rounded-lg disabled:opacity-50"
                >
                  ← Prev
                </button>
                <button
                  onClick={() => setCurrentImageIndex(Math.min(unlabeledImages.length - 1, currentImageIndex + 1))}
                  disabled={currentImageIndex === unlabeledImages.length - 1}
                  className="px-3 py-1 border border-gray-300 rounded-lg disabled:opacity-50"
                >
                  Next →
                </button>
              </div>
            </div>

            {isDrawingMode && (
              <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>Drawing Mode:</strong> Click and drag on the image to draw a bounding box around a person.
                </p>
              </div>
            )}

            <div ref={containerRef} className="border border-gray-300 rounded-lg overflow-hidden">
              <Stage
                ref={stageRef}
                width={dimensions.width}
                height={dimensions.height}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                className={isDrawingMode ? 'cursor-crosshair' : 'cursor-default'}
              >
                <Layer>
                  {imageObj && (
                    <KonvaImage
                      image={imageObj}
                      width={dimensions.width}
                      height={dimensions.height}
                    />
                  )}

                  {canvasBboxes.map((bbox, idx) => (
                    <Rect
                      key={bbox.id || idx}
                      x={bbox.x}
                      y={bbox.y}
                      width={bbox.width}
                      height={bbox.height}
                      stroke={bbox.isDrawing ? '#3b82f6' : (bbox.approved ? '#10b981' : '#ef4444')}
                      strokeWidth={bbox.isDrawing ? 3 : 2}
                      opacity={0.8}
                      dash={bbox.isDrawing ? [5, 5] : []}
                      onClick={() => !isDrawingMode && setSelectedBboxIndex(idx)}
                    />
                  ))}
                </Layer>
              </Stage>
            </div>

            <div className="mt-4 flex justify-end space-x-3">
              <button
                onClick={() => {
                  setCurrentImageIndex(Math.min(unlabeledImages.length - 1, currentImageIndex + 1));
                  setBboxes([]);
                }}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
              >
                Skip
              </button>
              <button
                onClick={approveAnnotation}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Approve & Save
              </button>
            </div>
          </div>

          {/* Detection List */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">
              Detections ({bboxes.filter(b => b.approved).length})
            </h2>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {bboxes.length === 0 && (
                <p className="text-gray-500 text-sm text-center py-8">
                  No detections. Run inference first.
                </p>
              )}

              {bboxes.map((bbox, idx) => (
                <div
                  key={idx}
                  className={`p-3 border rounded-lg cursor-pointer ${
                    selectedBboxIndex === idx ? 'border-primary-500 bg-primary-50' : 'border-gray-200'
                  }`}
                  onClick={() => setSelectedBboxIndex(idx)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center">
                        <span className="font-medium">Person #{idx + 1}</span>
                        {bbox.confidence && (
                          <span className="ml-2 text-xs text-gray-500">
                            {(bbox.confidence * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        x: {bbox.x1.toFixed(0)}, y: {bbox.y1.toFixed(0)}
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleBboxApproval(idx);
                        }}
                        className={`p-1 rounded ${
                          bbox.approved ? 'text-green-600' : 'text-gray-400'
                        }`}
                        title={bbox.approved ? 'Approved' : 'Denied'}
                      >
                        {bbox.approved ? '✓' : '✗'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteBbox(idx);
                        }}
                        className="p-1 text-red-600 hover:bg-red-50 rounded"
                        title="Delete"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* No Images */}
      {selectedCamera && unlabeledImages.length === 0 && (
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
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="text-gray-600">No unlabeled images found for this camera</p>
          <p className="text-sm text-gray-500 mt-2">Upload images from Camera Management page</p>
        </div>
      )}
    </div>
  );
}

export default AnnotationTool;
