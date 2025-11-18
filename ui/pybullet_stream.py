import cv2
import numpy as np
import pybullet as p
import pybullet_data
from flask import Flask, Response
import time

app = Flask(__name__)

# -------------------------
# Initialize PyBullet DIRECT
# -------------------------
cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0,0,-9.81)

# Load scene
p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Camera settings
cam_target = [0.6, 0, 0.3]
view = p.computeViewMatrix(
    cameraEyePosition=[1.2, 1.0, 0.8],
    cameraTargetPosition=cam_target,
    cameraUpVector=[0, 0, 1]
)

proj = p.computeProjectionMatrixFOV(
    fov=60, aspect=1.0,
    nearVal=0.1, farVal=10.0
)

def stream_frames():
    while True:
        p.stepSimulation()
        time.sleep(1/60)

        w, h, rgba, _, _ = p.getCameraImage(
            640, 480,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert PyBullet's nested tuple to NumPy array
        rgba = np.array(rgba)

        # If shape is (h, w, 4) good.
        # If PyBullet returns flat or weird format → reshape manually.
        if rgba.ndim == 1:
            rgba = rgba.reshape((h, w, 4))

        # Convert float64 → uint8 safely
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)

        # Now valid image. Convert to BGR for JPEG.
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue  # skip corrupted frames

        # Stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')


@app.route("/")
def video_feed():
    return Response(stream_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=5001, debug=False)
