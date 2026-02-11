
import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

function PipelineHeader() {
  const stages = [
    { label: 'Image Input', icon: '🖼️' },
    { label: 'Detection', icon: '🔍' },
    { label: 'Repair Planning', icon: '📝' },
    { label: 'Graph & Visualization', icon: '📊' },
    { label: 'Robotic Simulation', icon: '🤖' },
  ];
  return (
    <div className="pipeline-header">
      {stages.map((stage, idx) => (
        <div className="pipeline-stage" key={stage.label}>
          <span className="stage-icon">{stage.icon}</span>
          <span className="stage-label">{stage.label}</span>
          {idx < stages.length - 1 && <span className="stage-arrow">→</span>}
        </div>
      ))}
    </div>
  );
}


function App() {
  const [image, setImage] = useState(null);
  const [plan, setPlan] = useState('');
  const [damagedPart, setDamagedPart] = useState('');
  const [guide, setGuide] = useState('');
  const [simImage, setSimImage] = useState(null);
  const [simReady, setSimReady] = useState(false);
  const [loading, setLoading] = useState(false);

  const [simError, setSimError] = useState('');
  const [showStream, setShowStream] = useState(false);
  const [streamPlanPath, setStreamPlanPath] = useState('');
  
  // Camera defaults: show both robot and chair (dist 2.4, yaw 55°, pitch -25°)
  const [cameraDist, setCameraDist] = useState(2.4);
  const [cameraYaw, setCameraYaw] = useState(55.0);
  const [cameraPitch, setCameraPitch] = useState(-25.0);

  // Update camera parameters
  const updateCamera = useCallback(async (dist, yaw, pitch) => {
    try {
      const res = await fetch('/update-camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dist, yaw, pitch })
      });
      if (!res.ok) {
        console.error('Failed to update camera');
      }
    } catch (error) {
      console.error('Error updating camera:', error);
    }
  }, []);

  // Handle camera parameter changes with faster updates for real-time control
  useEffect(() => {
    // Update camera even if stream hasn't started yet (for when it does start)
    const timer = setTimeout(() => {
      updateCamera(cameraDist, cameraYaw, cameraPitch);
    }, 100); // Faster debounce 100ms for more responsive controls
    return () => clearTimeout(timer);
  }, [cameraDist, cameraYaw, cameraPitch, updateCamera]);

  // Note: Simulation now auto-starts after upload via the /upload endpoint
  // Removed auto-start on mount - simulation should only start after user uploads image

  // Live polling for simulation image - faster for real-time streaming
  useEffect(() => {
    if (!simReady || !showStream) return;
    
    setSimError('');
    let consecutiveErrors = 0;
    const maxErrors = 20; // Allow more time for simulation to start
    
    const id = setInterval(async () => {
      try {
        const res = await fetch(`/sim-stream.jpg?t=${Date.now()}`);
        if (res.ok) {
          setSimImage(`/sim-stream.jpg?t=${Date.now()}`);
          consecutiveErrors = 0; // Reset error count on success
          setSimError(''); // Clear any previous errors
        } else {
          consecutiveErrors++;
          // Only show error after many consecutive failures
          if (consecutiveErrors >= maxErrors) {
            setSimError('Simulation image not available. The simulation may still be starting...');
          }
        }
      } catch (error) {
        consecutiveErrors++;
        // Don't show error immediately, wait for multiple failures
        if (consecutiveErrors >= maxErrors) {
          setSimError('Unable to load simulation stream. Check if simulation is running.');
        }
      }
    }, 200); // Poll every 200ms for smoother real-time streaming (5 FPS)
    
    return () => clearInterval(id);
  }, [simReady, showStream]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!image) {
      setSimError('Please select an image first');
      return;
    }
    
    setLoading(true);
    setSimError('');
    setGuide('');
    setSimImage(null);
    setSimReady(false);
    setPlan('');
    setDamagedPart('');
    
    try {
      const fd = new FormData();
      fd.append('file', image);
      const res = await fetch('/upload', { method: 'POST', body: fd });
      
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || 'Upload failed');
      }
      
      const data = await res.json();
      setPlan(data.plan);
      setDamagedPart(data.damaged_part);
      setGuide(data.guide);
      
      // Set the plan path for streaming
      if (data.plan_path) {
        setStreamPlanPath(data.plan_path);
      } else if (data.plan_file) {
        // Construct full path if only filename provided
        setStreamPlanPath(`ui/uploads/${data.plan_file}`);
      } else {
        console.warn('No plan_path or plan_file in response');
      }
      
      // Auto-start simulation if server started it
      if (data.simulation_started) {
        setShowStream(true);
        setSimReady(true);
        console.log('Simulation auto-started by server');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setSimError(error.message || 'Failed to process image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <h1 className="main-title">🔧 Repair2Skill</h1>
        <PipelineHeader />
      </div>

      {/* Stage 1: Image Input - Full Width */}
      <section className="upload-section">
        <div className="upload-card">
          <h2>🖼️ 1. Image Input</h2>
          <form onSubmit={handleUpload} className="upload-form">
            <input type="file" onChange={e => setImage(e.target.files[0])} />
            <button type="submit" disabled={loading || !image} className="upload-btn">
              {loading ? '⏳ Processing…' : '▶️ Run Analysis'}
            </button>
          </form>
        </div>
      </section>

      {/* Two Column Layout */}
      <div className="main-content">
        {/* Left Column: Detection, Planning & Visual Guide */}
        <div className="left-column">
          {/* Stage 2 & 3: Detection & Repair Planning */}
          <section className="stage-card detection-card">
            <h2>🔍 2. Detection & 📝 3. Repair Planning</h2>
            {plan ? (
              <div className="plan-content">
                <div className="damaged-part-badge">
                  <span className="badge-label">Damaged Part:</span>
                  <span className="badge-value">{damagedPart || 'N/A'}</span>
                </div>
                <div className="plan-container">
                  <h4>Repair Plan:</h4>
                  <pre className="code-block">{plan}</pre>
                </div>
              </div>
            ) : (
              <div className="placeholder">Upload an image to see detection and repair plan.</div>
            )}
          </section>

          {/* Stage 4: Visual Repair Guide */}
          <section className="stage-card guide-card">
            <h2>🖼️ Visual Repair Guide</h2>
            {guide ? (
              <div className="guide-container">
                <img src={guide} alt="Visual Guide" className="guide-image" />
              </div>
            ) : (
              <div className="placeholder">Visual guide will appear here after analysis.</div>
            )}
          </section>
        </div>

        {/* Right Column: Robotic Simulation */}
        <div className="right-column">
          <section className="stage-card simulation-card">
            <h2>🤖 5. Robotic Simulation (Live Stream)</h2>
            {streamPlanPath && (
              <div className="plan-info">
                <span className="plan-label">Plan:</span>
                <span className="plan-file">{streamPlanPath.split(/[\\/]/).pop()}</span>
              </div>
            )}
            
            {/* Camera Controls */}
            <div className="camera-controls">
              <h3>📹 Camera Controls</h3>
              <div className="camera-grid">
                <div className="camera-control-item">
                  <label>
                    <span className="control-label">Zoom (Distance)</span>
                    <span className="control-value">{cameraDist.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="5.0"
                    step="0.1"
                    value={cameraDist}
                    onChange={(e) => setCameraDist(parseFloat(e.target.value))}
                    className="camera-slider"
                  />
                </div>
                <div className="camera-control-item">
                  <label>
                    <span className="control-label">Rotate (Yaw)</span>
                    <span className="control-value">{cameraYaw.toFixed(1)}°</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="360"
                    step="1"
                    value={cameraYaw}
                    onChange={(e) => setCameraYaw(parseFloat(e.target.value))}
                    className="camera-slider"
                  />
                </div>
                <div className="camera-control-item">
                  <label>
                    <span className="control-label">Angle (Pitch)</span>
                    <span className="control-value">{cameraPitch.toFixed(1)}°</span>
                  </label>
                  <input
                    type="range"
                    min="-90"
                    max="90"
                    step="1"
                    value={cameraPitch}
                    onChange={(e) => setCameraPitch(parseFloat(e.target.value))}
                    className="camera-slider"
                  />
                </div>
              </div>
            </div>
            
            {/* Simulation Stream */}
            <div className="simulation-view">
              {simError && (
                <div className="sim-error">{simError}</div>
              )}
              {simImage ? (
                <img 
                  src={`${simImage}?t=${Date.now()}`} 
                  alt="Live PyBullet Stream" 
                  className="sim-stream-image"
                  onError={(e) => {
                    console.warn('Image load error, will retry on next poll');
                  }}
                />
              ) : (
                <div className="sim-placeholder">
                  {simReady ? '⏳ Waiting for simulation to start...' : '🚀 Simulation will start automatically'}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;
