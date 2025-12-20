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
  const [simRunning, setSimRunning] = useState(false);

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
    const formData = new FormData();
    formData.append('file', image);
    try {
      const response = await fetch('http://localhost:3001/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setPlan(data.plan);
        setGuide(data.guide ? `http://localhost:3001${data.guide}` : null);
        setDamagedPart(data.damaged_part);
        setSimReady(true);
      }
    } catch (err) {
      setError('Failed to upload: ' + err.message);
    }
    setLoading(false);
  };

  const handleCameraChange = (param, value) => {
    setCamera((prev) => ({ ...prev, [param]: value }));
    // TODO: Send to backend if needed
  };

  const handleRunSimulation = async () => {
    setSimRunning(true);
    try {
      const response = await fetch('http://localhost:3001/simulate', {
        method: 'POST'
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        alert('Simulation completed! Check console for output.');
        console.log(data.output);
      }
    } catch (err) {
      setError('Failed to run simulation: ' + err.message);
    }
    setSimRunning(false);
  };

  return (
    <>
      <div className="header" style={{ background: 'linear-gradient(135deg, #1e90ff 0%, #00ced1 100%)', padding: '2rem 0', boxShadow: '0 8px 32px rgba(30,144,255,0.15)' }}>
        <h1 style={{ color: '#fff', fontSize: '3rem', fontWeight: 900, letterSpacing: '2px', textShadow: '0 4px 16px #1e90ff' }}>
          <span role="img" aria-label="wrench">🔧</span> Repair2Skill
        </h1>
        <p style={{ color: '#f8f9fa', fontSize: '1.2rem', fontWeight: 400, marginTop: '0.5rem' }}>
          Intelligent Furniture Damage Detection & <span style={{ color: '#ffc107', fontWeight: 700 }}>Repair Planning</span> System
        </p>
      </div>
      <div className="container" style={{ background: '#fff', borderRadius: '16px', boxShadow: '0 2px 16px rgba(30,144,255,0.10)', padding: '2rem', marginTop: '2rem' }}>
        {/* Left Panel */}
        <div className="panel" style={{ background: 'linear-gradient(135deg, #f8f9fa 0%, #e8ecff 100%)', borderRadius: '12px', boxShadow: '0 2px 8px #1e90ff22' }}>
          <h1 style={{ color: '#1e90ff', fontWeight: 700, fontSize: '2rem' }}>
            <span role="img" aria-label="camera">📸</span> Upload Broken Chair
          </h1>
          <div className="upload-section">
            <form onSubmit={handleUpload} id="uploadForm">
              <input type="file" accept="image/*" required onChange={handleImageChange} style={{ borderColor: '#1e90ff', background: '#f0f4ff' }} />
              <button type="submit" disabled={loading} style={{ background: 'linear-gradient(135deg, #1e90ff 0%, #00ced1 100%)', color: '#fff' }}>
                {loading ? <div className="spinner"></div> : '🚀 Run Analysis Pipeline'}
              </button>
            </form>
          </div>
          {damagedPart && (
            <div className="status-badge status-success" style={{ background: '#d4edda', color: '#155724', border: '1px solid #c3e6cb' }}>
              ✓ Damage detected on: <strong style={{ color: '#dc3545' }}>{damagedPart}</strong>
            </div>
          )}
          {error && (
            <div className="error-message" style={{ background: '#f8d7da', color: '#721c24', borderLeft: '4px solid #dc3545' }}>⚠️ {error}</div>
          )}
          <h3 style={{ color: '#2c3e50', fontWeight: 600, marginTop: '2rem' }}>
            <span role="img" aria-label="clipboard">📋</span> Visual Repair Guide
          </h3>
          {guide ? (
            <img src={guide} alt="Repair Guide" onError={e => e.target.style.display='none'} style={{ border: '2px solid #1e90ff', borderRadius: '10px', boxShadow: '0 2px 8px #1e90ff22', maxWidth: '100%', marginTop: '1rem' }} />
          ) : (
            <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
              <span role="img" aria-label="camera">📷</span> Upload an image to generate repair guide
            </p>
          )}
        </div>
        {/* Right Panel */}
        <div className="panel" style={{ background: 'linear-gradient(135deg, #f8f9fa 0%, #e8ecff 100%)', borderRadius: '12px', boxShadow: '0 2px 8px #1e90ff22' }}>
          <h1 style={{ color: '#00ced1', fontWeight: 700, fontSize: '2rem' }}>
            <span role="img" aria-label="tools">🛠️</span> Repair Plan & Simulation
          </h1>
          <h3 style={{ color: '#2c3e50', fontWeight: 600, marginTop: '2rem' }}>
            <span role="img" aria-label="notepad">📝</span> Generated Repair Plan
          </h3>
          {plan ? (
            plan.startsWith('Error') || plan.startsWith('No') ? (
              <div className="error-message" style={{ background: '#f8d7da', color: '#721c24', borderLeft: '4px solid #dc3545' }}>⚠️ {plan}</div>
            ) : (
              <div className="code-block" style={{ background: '#1e1e1e', color: '#00ff9d', border: '1px solid #333', borderRadius: '8px', padding: '18px', margin: '15px 0', fontFamily: 'Courier New, Monaco, monospace', fontSize: '13px', maxHeight: '350px', overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.5, boxShadow: 'inset 0 0 10px rgba(0,0,0,0.3)' }}>{plan}</div>
            )
          ) : (
            <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
              <span role="img" aria-label="chart">📊</span> Upload an image to generate repair plan
            </p>
          )}
          {/* Simulation Section */}
          {simReady ? (
            <div className="simulation-container" style={{ background: '#f8f9fa', borderRadius: '10px', border: '2px solid #e8e8e8', marginTop: '2rem', padding: '1rem' }}>
              <h3 style={{ color: '#2c3e50', fontWeight: 600 }}>
                <span role="img" aria-label="movie">🎬</span> Live Repair Simulation
              </h3>
              <button onClick={handleRunSimulation} disabled={simRunning} style={{ background: 'linear-gradient(135deg, #1e90ff 0%, #00ced1 100%)', color: '#fff', border: 'none', padding: '10px 20px', borderRadius: '5px', cursor: 'pointer', marginBottom: '1rem' }}>
                {simRunning ? 'Running Simulation...' : '▶️ Run Simulation'}
              </button>
              <img src="/static/simulation_placeholder.png" alt="Simulation Stream" style={{ background: '#000', border: '2px solid #333', borderRadius: '10px', maxWidth: '100%' }} />
              <div className="controls" style={{ marginTop: '1.5rem', padding: '1rem', background: '#fff', borderRadius: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', border: '1px solid #e8e8e8' }}>
                <div className="control-group" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <label style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px', color: '#1e90ff', textTransform: 'uppercase', letterSpacing: '0.5px' }}>🔍 Zoom (Distance)</label>
                  <input type="range" min="0.4" max="2.5" step="0.1" value={camera.dist} onChange={e => handleCameraChange('dist', e.target.value)} />
                </div>
                <div className="control-group" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <label style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px', color: '#00ced1', textTransform: 'uppercase', letterSpacing: '0.5px' }}>🔄 Rotate (Yaw)</label>
                  <input type="range" min="0" max="360" step="5" value={camera.yaw} onChange={e => handleCameraChange('yaw', e.target.value)} />
                </div>
                <div className="control-group" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <label style={{ fontSize: '13px', fontWeight: 600, marginBottom: '10px', color: '#ffc107', textTransform: 'uppercase', letterSpacing: '0.5px' }}>📐 Angle (Pitch)</label>
                  <input type="range" min="-89" max="-10" step="5" value={camera.pitch} onChange={e => handleCameraChange('pitch', e.target.value)} />
                </div>
              </div>
            </div>
          ) : (
            <>
              <h3 style={{ color: '#2c3e50', fontWeight: 600 }}>
                <span role="img" aria-label="movie">🎬</span> Live Repair Simulation
              </h3>
              <p style={{ color: '#999', textAlign: 'center', padding: '30px 0' }}>
                <span role="img" aria-label="hourglass">⏳</span> Simulation will appear here after analysis
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
