const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3002;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// Multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Ensure uploads directory exists
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

// Serve uploaded files
app.use('/uploads', express.static('uploads'));
// Serve visual guides from data/visual_guides
app.use('/visual_guides', express.static(path.join(__dirname, '../data/visual_guides')));

// SSE endpoint for streaming simulation output
app.post('/simulate-stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    let parsed;
    try {
      parsed = JSON.parse(body);
    } catch (e) {
      res.write(`event: error\ndata: Invalid JSON\n\n`);
      res.end();
      return;
    }
    const { plan, damagedPart, camera } = parsed;
    if (!plan || !damagedPart) {
      res.write(`event: error\ndata: Plan and damaged part are required\n\n`);
      res.end();
      return;
    }
    // Write plan to a temp file
    const planFilePath = path.resolve(__dirname, 'uploads', `plan_${Date.now()}.json`);
    fs.writeFileSync(planFilePath, plan);
    // Screenshot path
    const screenshotPath = path.resolve(__dirname, 'uploads', `simulation_${Date.now()}.png`);
    // Run the simulation
    const simProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', [
      'C:/Users/tripa/Downloads/PPT/Repair2Skill/pybullet_sim/run_simulation.py',
      '--plan', planFilePath,
      '--damaged-part', damagedPart,
      '--camera-dist', camera.dist,
      '--camera-yaw', camera.yaw,
      '--camera-pitch', camera.pitch,
      '--screenshot', screenshotPath
    ], {
      cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    simProcess.stdout.on('data', (data) => {
      res.write(`event: log\ndata: ${data.toString().replace(/\n/g, '\ndata: ')}\n\n`);
    });
    simProcess.stderr.on('data', (data) => {
      res.write(`event: error\ndata: ${data.toString().replace(/\n/g, '\ndata: ')}\n\n`);
    });
    simProcess.on('close', (code) => {
      if (code !== 0) {
        res.write(`event: error\ndata: Simulation failed with code ${code}\n\n`);
      } else {
        const screenshotUrl = `/uploads/simulation_${path.basename(screenshotPath, '.png')}.png`;
        res.write(`event: done\ndata: {\\"screenshot\\":\\"${screenshotUrl}\\"}\n\n`);
      }
      res.end();
    });
  });
});

// Upload endpoint
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const imagePath = path.resolve(req.file.path);

  // Run detect_damage.py
  const detectProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', [
    'C:/Users/tripa/Downloads/PPT/Repair2Skill/scripts/detect_damage.py',
    '--image', imagePath,
    '--threshold', '0.05'
  ], {
    cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
    stdio: ['pipe', 'pipe', 'pipe']
  });

  let detectionOutput = '';
  let detectionError = '';

  detectProcess.stdout.on('data', (data) => {
    detectionOutput += data.toString();
  });

  detectProcess.stderr.on('data', (data) => {
    detectionError += data.toString();
  });

  detectProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: `Detection failed: ${detectionError}` });
    }

    try {
      const detectionData = JSON.parse(detectionOutput.trim());
      const detectedPairs = detectionData.detected_pairs || [];
      if (detectedPairs.length === 0) {
        return res.status(400).json({ error: 'No damage detected' });
      }

      const pair = detectedPairs[0];
      const damagedPart = pair.part;
      const damageType = pair.damage_type;

      // Run generate_repair_plan.py
      const planProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', ['C:/Users/tripa/Downloads/PPT/Repair2Skill/scripts/generate_repair_plan.py', '--furniture', 'chair', '--part', damagedPart, '--damage', damageType], {
        cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let planOutput = '';
      let planError = '';

      planProcess.stdout.on('data', (data) => {
        planOutput += data.toString();
      });

      planProcess.stderr.on('data', (data) => {
        planError += data.toString();
      });

      planProcess.on('close', (planCode) => {
        if (planCode !== 0) {
          return res.status(500).json({ error: `Plan generation failed: ${planError}` });
        }

        try {
          const planData = JSON.parse(planOutput.trim());

          // Save plan to file
          const planFilePath = path.resolve(__dirname, 'uploads', `plan_${req.file.filename}.json`);
          fs.writeFileSync(planFilePath, JSON.stringify(planData, null, 2));

          // Generate visual guide
          const parts = [
            "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
          ];
          const partIdx = parts.indexOf(damagedPart);

          // Save guide in data/visual_guides
          const guidePath = path.resolve(__dirname, '../data/visual_guides', `guide_${req.file.filename}.png`);


          // Save detection JSON for visual guidance
          const detectionPath = path.resolve(__dirname, 'uploads', `detection_${req.file.filename}.json`);
          fs.writeFileSync(detectionPath, JSON.stringify(detectionData, null, 2));

          const guideProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', [
            'C:/Users/tripa/Downloads/PPT/Repair2Skill/scripts/render_visual_guidance.py',
            '--highlighted_part_idx', partIdx.toString(),
            '--save_path', guidePath,
            '--damage_report_path', detectionPath,
            '--plan_json_path', planFilePath
          ], {
            cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
            stdio: ['pipe', 'pipe', 'pipe']
          });

          let guideError = '';

          guideProcess.stderr.on('data', (data) => {
            guideError += data.toString();
          });


          guideProcess.on('close', (guideCode) => {
            const guideUrl = guideCode === 0 ? `/visual_guides/guide_${req.file.filename}.png` : null;

            res.json({
              plan: JSON.stringify(planData, null, 2),
              damaged_part: damagedPart,
              guide: guideUrl
            });
          });

        } catch (e) {
          res.status(500).json({ error: `Failed to parse plan: ${e.message}` });
        }
      });

    } catch (e) {
      res.status(500).json({ error: `Failed to parse detection: ${e.message}` });
    }
  });
});

// Simulation endpoint
app.post('/simulate', (req, res) => {
  const { plan, damagedPart, camera } = req.body;
  if (!plan || !damagedPart) {
    return res.status(400).json({ error: 'Plan and damaged part are required' });
  }

  // Write plan to a temp file
  const planFilePath = path.resolve(__dirname, 'uploads', `plan_${Date.now()}.json`);
  fs.writeFileSync(planFilePath, plan);

  // Screenshot path
  const screenshotPath = path.resolve(__dirname, 'uploads', `simulation_${Date.now()}.png`);

  // Run the simulation
  const simProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', [
    'C:/Users/tripa/Downloads/PPT/Repair2Skill/pybullet_sim/run_simulation.py',
    '--plan', planFilePath,
    '--damaged-part', damagedPart,
    '--camera-dist', camera.dist,
    '--camera-yaw', camera.yaw,
    '--camera-pitch', camera.pitch,
    '--screenshot', screenshotPath
  ], {
    cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
    stdio: ['pipe', 'pipe', 'pipe']
  });

  let simOutput = '';
  let simError = '';

  simProcess.stdout.on('data', (data) => {
    simOutput += data.toString();
  });

  simProcess.stderr.on('data', (data) => {
    simError += data.toString();
  });

  simProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: `Simulation failed: ${simError}` });
    }

    const screenshotUrl = `/uploads/simulation_${path.basename(screenshotPath, '.png')}.png`;
    res.json({ message: 'Simulation completed', output: simOutput, screenshot: screenshotUrl });
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});