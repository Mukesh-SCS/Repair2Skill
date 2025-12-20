const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3001;

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

// Upload endpoint
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const imagePath = path.resolve(req.file.path);

  // Run detect_damage.py
  const detectProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', ['C:/Users/tripa/Downloads/PPT/Repair2Skill/scripts/detect_damage.py', imagePath], {
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

          // Generate visual guide
          const parts = [
            "seat", "back", "front_left_leg", "front_right_leg",
            "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
          ];
          const partIdx = parts.indexOf(damagedPart);
          const guidePath = path.join('uploads', `guide_${req.file.filename}.png`);

          const guideProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', ['C:/Users/tripa/Downloads/PPT/Repair2Skill/scripts/render_visual_guidance.py', '--highlighted_part_idx', partIdx.toString(), '--save_path', guidePath], {
            cwd: 'C:/Users/tripa/Downloads/PPT/Repair2Skill',
            stdio: ['pipe', 'pipe', 'pipe']
          });

          let guideError = '';

          guideProcess.stderr.on('data', (data) => {
            guideError += data.toString();
          });

          guideProcess.on('close', (guideCode) => {
            const guideUrl = guideCode === 0 ? `/uploads/guide_${req.file.filename}.png` : null;

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
  // Run the simulation
  const simProcess = spawn('C:/Users/tripa/Downloads/PPT/Repair2Skill/.venv/Scripts/python.exe', ['C:/Users/tripa/Downloads/PPT/Repair2Skill/pybullet_sim/run_simulation.py'], {
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

    res.json({ message: 'Simulation completed', output: simOutput });
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});