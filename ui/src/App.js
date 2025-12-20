
import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [plan, setPlan] = useState('');
  const [guide, setGuide] = useState(null);
  const [damagedPart, setDamagedPart] = useState('');
  const [loading, setLoading] = useState(false);
  const [simReady, setSimReady] = useState(false);
  const [error, setError] = useState('');
  const [camera, setCamera] = useState({ dist: 1.0, yaw: 50, pitch: -25 });

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setError('');
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    setError('');
    // TODO: Replace with backend API call
    setTimeout(() => {
      setPlan('{\n  "repair_sequence": [\n    { "step": "Remove seat" }, { "step": "Replace seat" }\n  ]\n}');
      setGuide('/static/seat_repair_guide.png');
      setDamagedPart('seat');
      setSimReady(true);
      setLoading(false);
    }, 1200);
  };

  const handleCameraChange = (param, value) => {
    setCamera((prev) => ({ ...prev, [param]: value }));
    // TODO: Send to backend if needed
  };

  return (
    <>
      <div className="header">
        <h1>🔧 Repair2Skill</h1>
        <p>Intelligent Furniture Damage Detection & Repair Planning System</p>
      </div>
      <div className="container">
        {/* Left Panel */}
        <div className="panel">
          <h1>📸 Upload Broken Chair</h1>
          <div className="upload-section">
            <form onSubmit={handleUpload} id="uploadForm">
              <input type="file" accept="image/*" required onChange={handleImageChange} />
              <button type="submit" disabled={loading}>
                {loading ? <div className="spinner"></div> : '🚀 Run Analysis Pipeline'}
              </button>
            </form>
          </div>
          {damagedPart && (
            <div className="status-badge status-success">
              ✓ Damage detected on: <strong>{damagedPart}</strong>
            </div>
          )}
          {error && (
            <div className="error-message">⚠️ {error}</div>
          )}
          <h3>📋 Visual Repair Guide</h3>
          {guide ? (
            <img src={guide} alt="Repair Guide" onError={e => e.target.style.display='none'} />
          ) : (
            <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
              📷 Upload an image to generate repair guide
            </p>
          )}
        </div>
        {/* Right Panel */}
        <div className="panel">
          <h1>🛠️ Repair Plan & Simulation</h1>
          <h3>📝 Generated Repair Plan</h3>
          {plan ? (
            plan.startsWith('Error') || plan.startsWith('No') ? (
              <div className="error-message">⚠️ {plan}</div>
            ) : (
              <div className="code-block">{plan}</div>
            )
          ) : (
            <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
              📊 Upload an image to generate repair plan
            </p>
          )}
          {/* Simulation Section */}
          {simReady ? (
            <div className="simulation-container">
              <h3>🎬 Live Repair Simulation</h3>
              <img src="/static/simulation_placeholder.png" alt="Simulation Stream" style={{ background: '#000' }} />
              <div className="controls">
                <div className="control-group">
                  <label>🔍 Zoom (Distance)</label>
                  <input type="range" min="0.4" max="2.5" step="0.1" value={camera.dist} onChange={e => handleCameraChange('dist', e.target.value)} />
                </div>
                <div className="control-group">
                  <label>🔄 Rotate (Yaw)</label>
                  <input type="range" min="0" max="360" step="5" value={camera.yaw} onChange={e => handleCameraChange('yaw', e.target.value)} />
                </div>
                <div className="control-group">
                  <label>📐 Angle (Pitch)</label>
                  <input type="range" min="-89" max="-10" step="5" value={camera.pitch} onChange={e => handleCameraChange('pitch', e.target.value)} />
                </div>
              </div>
            </div>
          ) : (
            <h3>🎬 Live Repair Simulation</h3>
            <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
              ⏳ Simulation will appear here after analysis
            </p>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
