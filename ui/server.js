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
  const options = {
    hostname: 'localhost',
    port: 8080,
    path: '/frame.jpg',
    method: 'GET'
  };
  
  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  
  proxyReq.on('error', (e) => {
    console.error(`[SERVER] Stream proxy error: ${e.message}`);
    res.status(503).json({ error: 'Simulation stream not available' });
  });
  
  proxyReq.end();
});

/* ===============================
   MJPEG streaming endpoint: launches PyBullet GUI and saves periodic screenshots
   =============================== */
let simProcess = null;
let screenshotInterval = null;

// Store current camera parameters - set to user's preferred defaults
let currentCamera = { dist: 1.70, yaw: 180.0, pitch: 9.0 };

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
    console.error(`[DEFAULT-SIM ERROR] ${data.toString().trim()}`);
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
    console.error(`[SIM ERROR] ${data.toString().trim()}`);
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
  const detect = spawn(PYTHON, [
    `${ROOT}/scripts/detect_damage.py`,
    '--image', imagePath,
    '--threshold', '0.05'
  ], { cwd: ROOT });

  let detectOut = '';
  let detectErr = '';

  detect.stdout.on('data', d => detectOut += d.toString());
  detect.stderr.on('data', d => {
    detectErr += d.toString();
    console.error(`[DETECT ERROR] ${d.toString().trim()}`);
  });

  detect.on('close', (code) => {
    if (code !== 0) {
      console.error(`[SERVER] Detection failed with code ${code}`);
      console.error(`[SERVER] Error output: ${detectErr}`);
      return res.status(500).json({ error: `Detection failed: ${detectErr || 'Unknown error'}` });
    }

    let detection;
    try {
      detection = JSON.parse(detectOut.trim());
    } catch (e) {
      console.error(`[SERVER] Failed to parse detection output: ${e}`);
      console.error(`[SERVER] Raw output: ${detectOut}`);
      return res.status(500).json({ error: 'Failed to parse detection output', details: detectOut });
    }

    const pair = detection.detected_pairs?.[0];
    if (!pair) {
      return res.status(400).json({ error: 'No damage detected in image' });
    }

    const damagedPart = pair.part;
    const damageType = pair.damage_type;

    console.log(`[SERVER] Detected damage: ${damagedPart} (${damageType})`);

    /* ---- Stage 3: Repair Planning ---- */
    const planProc = spawn(PYTHON, [
      `${ROOT}/scripts/generate_repair_plan.py`,
      '--furniture', 'chair',
      '--part', damagedPart,
      '--damage', damageType
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
        console.error(`[SERVER] Plan generation failed with code ${code}`);
        console.error(`[SERVER] Error output: ${planErr}`);
        return res.status(500).json({ error: `Plan generation failed: ${planErr || 'Unknown error'}` });
      }

      let plan;
      try {
        plan = JSON.parse(planOut.trim());
      } catch (e) {
        console.error(`[SERVER] Failed to parse plan output: ${e}`);
        console.error(`[SERVER] Raw output: ${planOut}`);
        return res.status(500).json({ error: 'Failed to parse plan output', details: planOut });
      }

      const planFile = `repair_plan_${Date.now()}.json`;
      const planPath = path.join(UPLOAD_DIR, planFile);
      fs.writeFileSync(planPath, JSON.stringify(plan, null, 2));
      console.log(`[SERVER] Plan saved to: ${planPath}`);

      /* ---- Stage 4: Visual Guide ---- */
      const detectionPath = path.join(UPLOAD_DIR, `detection_${Date.now()}.json`);
      fs.writeFileSync(detectionPath, JSON.stringify(detection, null, 2));

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
            
            // Save initial camera params
            fs.writeFileSync(cameraParamsPath, JSON.stringify(currentCamera));
            console.log(`[SERVER] Camera params saved to: ${cameraParamsPath}`);
            
            // Start simulation automatically
            console.log(`[SERVER] Auto-starting simulation with plan: ${planPath}`);
            console.log(`[SERVER] Screenshot will be saved to: ${screenshotPath}`);
            console.log(`[SERVER] Damaged part: ${damagedPart}`);
            
            // Use absolute paths for better compatibility
            const absPlanPath = path.resolve(planPath);
            const absScreenshotPath = path.resolve(screenshotPath);
            const absCameraParamsPath = path.resolve(cameraParamsPath);
            
            const autoSimProcess = spawn(PYTHON, [
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

            let autoSimOutput = '';
            autoSimProcess.stdout.on('data', (data) => {
              autoSimOutput += data.toString();
              console.log(`[AUTO-SIM] ${data.toString().trim()}`);
            });

            autoSimProcess.stderr.on('data', (data) => {
              console.error(`[AUTO-SIM ERROR] ${data.toString().trim()}`);
            });

            autoSimProcess.on('close', (code) => {
              console.log(`[SERVER] Auto-simulation process exited with code ${code}`);
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
