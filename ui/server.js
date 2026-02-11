const express = require('express');
const multer = require('multer');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3002;

/* ===============================
   Middleware
   =============================== */
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  next();
});

/* ===============================
   Paths & Setup
   =============================== */
const ROOT = path.resolve(__dirname, '..');
const PYTHON = process.env.PYTHON_PATH || `${ROOT}/.venv/Scripts/python.exe`;

const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/visual_guides', express.static(path.join(ROOT, 'data/visual_guides')));

const upload = multer({ dest: UPLOAD_DIR });

/* ===============================
   Live Simulation Image (Direct Streaming Proxy)
   =============================== */
app.get('/sim-stream.jpg', (req, res) => {
  // Proxy to Python streaming server for direct frame access
  const http = require('http');
  
  // Check if simulation is running (optional check - don't block if check fails)
  if (simProcess && simProcess.killed) {
    // Simulation was killed, return placeholder
    res.status(204).end();
    return;
  }
  
  const options = {
    hostname: 'localhost',
    port: 8080,
    path: '/frame.jpg',
    method: 'GET',
    timeout: 2000 // 2 second timeout
  };
  
  const proxyReq = http.request(options, (proxyRes) => {
    // Copy headers but ensure proper content type
    const headers = { ...proxyRes.headers };
    headers['Cache-Control'] = 'no-cache, no-store, must-revalidate';
    headers['Pragma'] = 'no-cache';
    headers['Expires'] = '0';
    
    res.writeHead(proxyRes.statusCode, headers);
    proxyRes.pipe(res);
  });
  
  proxyReq.on('error', (e) => {
    // Only log unexpected errors (not connection refused which is normal when server isn't running)
    if (e.code !== 'ECONNREFUSED' && e.code !== 'ETIMEDOUT') {
      console.error(`[SERVER] Stream proxy error: ${e.message}`);
    }
    
    // Return a 1x1 transparent PNG instead of JSON (browsers expect image)
    // This prevents broken image icons in the UI
    const transparentPixel = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      'base64'
    );
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': transparentPixel.length,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0'
    });
    res.end(transparentPixel);
  });
  
  proxyReq.on('timeout', () => {
    proxyReq.destroy();
    const transparentPixel = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      'base64'
    );
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': transparentPixel.length,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0'
    });
    res.end(transparentPixel);
  });
  
  proxyReq.end();
});

/* ===============================
   MJPEG streaming endpoint: launches PyBullet GUI and saves periodic screenshots
   =============================== */
let simProcess = null;
let screenshotInterval = null;

// Default camera: show both robot (left) and chair (right). dist 2.4, yaw 55°, pitch -25°
let currentCamera = { dist: 2.4, yaw: 55.0, pitch: -25.0 };

// Endpoint to update camera parameters
app.post('/update-camera', (req, res) => {
  const { dist, yaw, pitch } = req.body;
  if (dist !== undefined) currentCamera.dist = parseFloat(dist);
  if (yaw !== undefined) currentCamera.yaw = parseFloat(yaw);
  if (pitch !== undefined) currentCamera.pitch = parseFloat(pitch);
  
  // Save to file so Python process can read it
  const cameraPath = path.join(UPLOAD_DIR, 'camera_params.json');
  fs.writeFileSync(cameraPath, JSON.stringify(currentCamera));
  
  console.log(`[SERVER] Camera updated:`, currentCamera);
  res.json({ success: true, camera: currentCamera });
});

// Endpoint to get current camera parameters
app.get('/camera-params', (req, res) => {
  res.json(currentCamera);
});

// Endpoint to start simulation with default/demo plan
app.post('/start-default-sim', (req, res) => {
  // Use a default plan file or create a demo plan
  const defaultPlanPath = path.join(ROOT, 'outputs', 'repair_plan_back_left_leg_broken.json');
  
  // If default plan doesn't exist, try to find any plan in outputs
  let planPath = defaultPlanPath;
  if (!fs.existsSync(planPath)) {
    // Look for any repair plan in outputs directory
    const outputsDir = path.join(ROOT, 'outputs');
    if (fs.existsSync(outputsDir)) {
      const files = fs.readdirSync(outputsDir);
      const planFile = files.find(f => f.startsWith('repair_plan_') && f.endsWith('.json'));
      if (planFile) {
        planPath = path.join(outputsDir, planFile);
      }
    }
  }
  
  // If still no plan found, create a simple demo plan
  if (!fs.existsSync(planPath)) {
    const demoPlan = {
      repair_sequence: [
        {
          step_id: 1,
          action_type: "inspect",
          target_part: "back_left_leg",
          description: "Inspect the damaged part"
        },
        {
          step_id: 2,
          action_type: "tighten",
          target_part: "back_left_leg",
          description: "Tighten the loose part"
        }
      ]
    };
    planPath = path.join(UPLOAD_DIR, 'demo_repair_plan.json');
    fs.writeFileSync(planPath, JSON.stringify(demoPlan, null, 2));
    console.log(`[SERVER] Created demo plan at: ${planPath}`);
  }
  
  // Update camera from request
  if (req.body.camera) {
    currentCamera = {
      dist: req.body.camera.dist || currentCamera.dist,
      yaw: req.body.camera.yaw || currentCamera.yaw,
      pitch: req.body.camera.pitch || currentCamera.pitch
    };
  }
  
  const cameraParamsPath = path.join(UPLOAD_DIR, 'camera_params.json');
  const screenshotPath = path.join(UPLOAD_DIR, 'simulation_latest.jpg');
  
  // Ensure uploads directory exists
  if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
  }
  
  // Save camera params
  fs.writeFileSync(cameraParamsPath, JSON.stringify(currentCamera));
  console.log(`[SERVER] Starting default simulation with plan: ${planPath}`);
  
  // Extract damaged part
  let damagedPart = 'back_left_leg';
  try {
    if (fs.existsSync(planPath)) {
      const planData = JSON.parse(fs.readFileSync(planPath, 'utf8'));
      const firstStep = planData.repair_sequence?.[0];
      if (firstStep && firstStep.target_part) {
        damagedPart = firstStep.target_part;
      }
    }
  } catch (e) {
    console.warn(`[SERVER] Could not extract damaged part: ${e.message}`);
  }
  
  // Kill any previous simulation
  if (simProcess) {
    try {
      simProcess.kill();
    } catch (e) {
      console.error('Error killing previous sim process:', e);
    }
  }
  
  // Use absolute paths
  const absPlanPath = path.resolve(planPath);
  const absScreenshotPath = path.resolve(screenshotPath);
  const absCameraParamsPath = path.resolve(cameraParamsPath);
  
  simProcess = spawn(PYTHON, [
    `${ROOT}/pybullet_sim/run_simulation.py`,
    '--plan', absPlanPath,
    '--damaged-part', damagedPart,
    '--stream-port', '8080',  // Use direct streaming instead of screenshots
    '--camera-params', absCameraParamsPath,
    '--camera-dist', currentCamera.dist.toString(),
    '--camera-yaw', currentCamera.yaw.toString(),
    '--camera-pitch', currentCamera.pitch.toString()
  ], { 
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  let simOutput = '';
  let simError = '';

  simProcess.stdout.on('data', (data) => {
    simOutput += data.toString();
    console.log(`[DEFAULT-SIM] ${data.toString().trim()}`);
  });

  simProcess.stderr.on('data', (data) => {
    simError += data.toString();
    console.log(`[DEFAULT-SIM] ${data.toString().trim()}`);
  });

  simProcess.on('close', (code) => {
    console.log(`[SERVER] Default simulation process exited with code ${code}`);
  });

  res.json({ 
    success: true,
    message: 'Default simulation started',
    planPath: planPath
  });
});

app.post('/start-sim-stream', (req, res) => {
  const planPath = req.body.planPath;
  if (!planPath) return res.status(400).json({ error: 'planPath required' });

  // Kill any previous processes
  if (simProcess) {
    try {
      simProcess.kill();
    } catch (e) {
      console.error('Error killing sim process:', e);
    }
  }
  if (screenshotInterval) {
    clearInterval(screenshotInterval);
    screenshotInterval = null;
  }

  // Extract damaged part from plan file
  let damagedPart = 'back_left_leg'; // default fallback
  try {
    if (fs.existsSync(planPath)) {
      const planData = JSON.parse(fs.readFileSync(planPath, 'utf8'));
      const firstStep = planData.repair_sequence?.[0];
      if (firstStep && firstStep.target_part) {
        damagedPart = firstStep.target_part;
      }
    }
  } catch (e) {
    console.warn(`[SERVER] Could not extract damaged part from plan: ${e.message}`);
  }

  // Update camera from request or use current
  if (req.body.camera) {
    currentCamera = {
      dist: req.body.camera.dist || currentCamera.dist,
      yaw: req.body.camera.yaw || currentCamera.yaw,
      pitch: req.body.camera.pitch || currentCamera.pitch
    };
    const cameraPath = path.join(UPLOAD_DIR, 'camera_params.json');
    fs.writeFileSync(cameraPath, JSON.stringify(currentCamera));
  }

  // Launch PyBullet simulation in HEADLESS mode (no GUI window)
  console.log(`[SERVER] Starting headless simulation with plan: ${planPath}`);
  console.log(`[SERVER] Detected damaged part: ${damagedPart}`);
  console.log(`[SERVER] Camera:`, currentCamera);
  
  const screenshotPath = path.join(UPLOAD_DIR, 'simulation_latest.jpg');
  const cameraParamsPath = path.join(UPLOAD_DIR, 'camera_params.json');
  
  simProcess = spawn(PYTHON, [
    `${ROOT}/pybullet_sim/run_simulation.py`,
    '--plan', planPath,
    '--damaged-part', damagedPart,
    '--screenshot', screenshotPath,
    '--camera-params', cameraParamsPath
    // Note: NOT using --gui flag, so it runs headless
  ], { 
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  let simOutput = '';
  let simError = '';

  simProcess.stdout.on('data', (data) => {
    simOutput += data.toString();
    console.log(`[SIM] ${data.toString().trim()}`);
  });

  simProcess.stderr.on('data', (data) => {
    simError += data.toString();
    console.log(`[SIM] ${data.toString().trim()}`);
  });

  simProcess.on('close', (code) => {
    console.log(`[SERVER] Simulation process exited with code ${code}`);
    if (screenshotInterval) {
      clearInterval(screenshotInterval);
      screenshotInterval = null;
    }
  });

  // Note: The simulation script itself handles periodic screenshot saving
  // No need for a separate interval here since the Python script does it

  res.json({ 
    message: 'Simulation started. Screenshots will be saved periodically.',
    planPath: planPath
  });
});

/* ===============================
   Upload → Detect → Plan → Visual Guide → Sim
   =============================== */
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const imagePath = path.resolve(req.file.path);

  /* ---- Camera (from UI or defaults) ---- */
  const camera = req.body.camera
    ? JSON.parse(req.body.camera)
    : { dist: 1.8, yaw: 40, pitch: -35 };

  console.log(`[SERVER] Processing image: ${imagePath}`);

  /* ---- Stage 2: Damage Detection ---- */
  // Use lower threshold and enable debug mode for better detection
  const detect = spawn(PYTHON, [
    `${ROOT}/scripts/detect_damage.py`,
    '--image', imagePath,
    '--threshold', '0.05',  // Lower threshold for better detection
    '--debug'  // Enable debug output to see what's being detected
  ], { cwd: ROOT });

  let detectOut = '';
  let detectErr = '';

  detect.stdout.on('data', d => {
    detectOut += d.toString();
    console.log(`[DETECT] ${d.toString().trim()}`);
  });
  detect.stderr.on('data', d => {
    detectErr += d.toString();
    console.error(`[DETECT ERROR] ${d.toString().trim()}`);
  });

  detect.on('close', (code) => {
    if (code !== 0) {
      console.error(`[SERVER] Detection failed with code ${code}`);
      console.error(`[SERVER] Error output: ${detectErr}`);
      console.error(`[SERVER] Stdout output: ${detectOut}`);
      return res.status(500).json({ 
        error: `Detection failed: ${detectErr || 'Unknown error'}`,
        details: detectErr,
        stdout: detectOut
      });
    }

    let detection;
    try {
      // Debug output now goes to stderr, so stdout should only contain JSON
      // But handle multi-line JSON (pretty-printed) by finding the JSON block
      const trimmed = detectOut.trim();
      
      // Try to find JSON object (could be single line or multi-line)
      let jsonStart = trimmed.indexOf('{');
      if (jsonStart === -1) {
        throw new Error('No JSON found in output');
      }
      
      // Extract JSON from the first { to the last }
      let jsonEnd = trimmed.lastIndexOf('}');
      if (jsonEnd === -1 || jsonEnd < jsonStart) {
        throw new Error('Invalid JSON format');
      }
      
      const jsonStr = trimmed.substring(jsonStart, jsonEnd + 1);
      detection = JSON.parse(jsonStr);
    } catch (e) {
      console.error(`[SERVER] Failed to parse detection output: ${e}`);
      console.error(`[SERVER] Raw stdout: ${detectOut}`);
      console.error(`[SERVER] Raw stderr: ${detectErr}`);
      return res.status(500).json({ 
        error: 'Failed to parse detection output', 
        details: detectOut,
        stderr: detectErr,
        parseError: e.message
      });
    }

    console.log(`[SERVER] Detection result:`, JSON.stringify(detection, null, 2));
    
    const pairs = detection.detected_pairs || [];
    if (!pairs || pairs.length === 0) {
      console.warn(`[SERVER] No damage detected. Detection result:`, detection);
      return res.status(400).json({ 
        error: 'No damage detected in image',
        details: 'The model did not detect any damage. This could mean: 1) The image does not contain visible damage, 2) The model needs retraining, 3) Try a different image with more obvious damage.',
        detection_result: detection
      });
    }

    const ts = Date.now();
    const detectionPath = path.join(UPLOAD_DIR, `detection_${ts}.json`);
    const planFile = `repair_plan_${ts}.json`;
    const planPath = path.join(UPLOAD_DIR, planFile);
    const damageReportPath = path.join(UPLOAD_DIR, `damage_report_${ts}.json`);
    fs.writeFileSync(detectionPath, JSON.stringify(detection, null, 2));

    /* ---- Stage 3: Damage report + deterministic plan + validate (Steps 1–3) ---- */
    const planProc = spawn(PYTHON, [
      `${ROOT}/scripts/build_repair_plan_from_detection.py`,
      '--detection', path.resolve(detectionPath),
      '--plan', path.resolve(planPath),
      '--damage-report', path.resolve(damageReportPath)
    ], { cwd: ROOT });

    let planOut = '';
    let planErr = '';

    planProc.stdout.on('data', d => planOut += d.toString());
    planProc.stderr.on('data', d => {
      planErr += d.toString();
      console.error(`[PLAN ERROR] ${d.toString().trim()}`);
    });

    planProc.on('close', (code) => {
      if (code !== 0) {
        console.error(`[SERVER] Plan build failed with code ${code}`);
        console.error(`[SERVER] Error output: ${planErr}`);
        let errMsg = planErr || 'Unknown error';
        try {
          const lastLine = planOut.trim().split('\n').pop();
          const j = JSON.parse(lastLine);
          if (j.error) errMsg = j.error;
        } catch (_) {}
        return res.status(500).json({ error: `Plan build failed: ${errMsg}` });
      }

      let result;
      try {
        const lastLine = planOut.trim().split('\n').pop();
        result = JSON.parse(lastLine);
      } catch (e) {
        console.error(`[SERVER] Failed to parse plan script output: ${e}`);
        console.error(`[SERVER] Raw output: ${planOut}`);
        return res.status(500).json({ error: 'Failed to parse plan script output', details: planOut });
      }

      if (result.error) {
        return res.status(400).json({ error: result.error, details: result });
      }

      const damagedPart = result.damaged_part;
      const planPathResolved = result.plan_path || planPath;
      console.log(`[SERVER] Plan saved to: ${planPathResolved}, damaged part: ${damagedPart}`);

      let plan;
      try {
        plan = JSON.parse(fs.readFileSync(planPathResolved, 'utf8'));
      } catch (_) {
        plan = { repair_sequence: [] };
      }

      /* ---- Stage 4: Visual Guide (uses detection for detected_pairs) ---- */

      const guideName = `guide_${Date.now()}.png`;
      const guidePath = path.join(ROOT, 'data/visual_guides', guideName);
      
      // Ensure directory exists
      const guideDir = path.dirname(guidePath);
      if (!fs.existsSync(guideDir)) {
        fs.mkdirSync(guideDir, { recursive: true });
      }

      const guideProc = spawn(PYTHON, [
        `${ROOT}/scripts/render_visual_guidance.py`,
        '--save_path', guidePath,
        '--damage_report_path', detectionPath,
        '--plan_json_path', planPath
      ], { cwd: ROOT });

      let guideErr = '';
      guideProc.stderr.on('data', d => {
        guideErr += d.toString();
        console.error(`[GUIDE ERROR] ${d.toString().trim()}`);
      });

      guideProc.on('close', (code) => {
        // Wait for the guide image to exist (max 3s)
        let waited = 0;
        const waitForGuide = setInterval(() => {
          if (fs.existsSync(guidePath) || waited > 3000) {
            clearInterval(waitForGuide);
            
            if (!fs.existsSync(guidePath)) {
              console.warn(`[SERVER] Guide image not created after 3s: ${guidePath}`);
            }

            // Auto-start simulation after visual guide is ready
            const cameraParamsPath = path.join(UPLOAD_DIR, 'camera_params.json');
            const screenshotPath = path.join(UPLOAD_DIR, 'simulation_latest.jpg');
            
            // Ensure uploads directory exists
            if (!fs.existsSync(UPLOAD_DIR)) {
              fs.mkdirSync(UPLOAD_DIR, { recursive: true });
            }
            
            // Kill any previous simulation before starting a new one
            if (simProcess) {
              try {
                simProcess.kill();
                console.log('[SERVER] Killed previous simulation process');
              } catch (e) {
                console.error('[SERVER] Error killing previous sim process:', e);
              }
            }
            
            // Save initial camera params
            fs.writeFileSync(cameraParamsPath, JSON.stringify(currentCamera));
            console.log(`[SERVER] Camera params saved to: ${cameraParamsPath}`);
            
            // Start simulation automatically (one sim per session; previous killed above)
            console.log(`[SERVER] Auto-starting simulation with plan: ${planPathResolved}`);
            console.log(`[SERVER] Screenshot will be saved to: ${screenshotPath}`);
            console.log(`[SERVER] Damaged part: ${damagedPart}`);
            
            // Use absolute paths for better compatibility
            const absPlanPath = path.resolve(planPathResolved);
            const absScreenshotPath = path.resolve(screenshotPath);
            const absCameraParamsPath = path.resolve(cameraParamsPath);
            
            // Use global simProcess to track and manage the simulation
            simProcess = spawn(PYTHON, [
              `${ROOT}/pybullet_sim/run_simulation.py`,
              '--plan', absPlanPath,
              '--damaged-part', damagedPart,
              '--stream-port', '8080',  // Use direct streaming instead of screenshots
              '--camera-params', absCameraParamsPath,
              '--camera-dist', currentCamera.dist.toString(),
              '--camera-yaw', currentCamera.yaw.toString(),
              '--camera-pitch', currentCamera.pitch.toString()
            ], { 
              cwd: ROOT,
              stdio: ['ignore', 'pipe', 'pipe']
            });

            let simOutput = '';
            simProcess.stdout.on('data', (data) => {
              simOutput += data.toString();
              console.log(`[AUTO-SIM] ${data.toString().trim()}`);
            });

            simProcess.stderr.on('data', (data) => {
              console.log(`[AUTO-SIM] ${data.toString().trim()}`);
            });

            simProcess.on('close', (code) => {
              console.log(`[SERVER] Auto-simulation process exited with code ${code}`);
              simProcess = null; // Clear the process reference
            });

            // Return response with all data
            res.json({
              plan: JSON.stringify(plan, null, 2),
              damaged_part: damagedPart,
              guide: `/visual_guides/${guideName}`,
              plan_path: planPath,
              plan_file: planFile,
              simulation_started: true
            });
          }
          waited += 100;
        }, 100);
      });
    });
  });
});

/* ===============================
   Start Server
   =============================== */
app.listen(port, () => {
  console.log(`[SERVER] Server running at http://localhost:${port}`);
  console.log(`[SERVER] ROOT: ${ROOT}`);
  console.log(`[SERVER] PYTHON: ${PYTHON}`);
  console.log(`[SERVER] UPLOAD_DIR: ${UPLOAD_DIR}`);
});
