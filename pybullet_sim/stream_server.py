"""
HTTP server for streaming PyBullet simulation frames directly.
No disk I/O - streams frames directly from PyBullet's getCameraImage.
"""

import http.server
import socketserver
import threading
import json
import os
import time
import pybullet as p
import numpy as np
from PIL import Image
import io
import math

class StreamingHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for streaming simulation frames."""
    
    def do_GET(self):
        if self.path == '/stream':
            # MJPEG stream
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            try:
                while True:
                    frame = self.server.get_frame()
                    if frame:
                        self.wfile.write(b'--jpgboundary\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(1.0 / 30)  # 30 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected
                
        elif self.path == '/frame.jpg':
            # Single frame
            frame = self.server.get_frame()
            if frame:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


class StreamingServer(socketserver.TCPServer):
    """HTTP server that streams PyBullet frames."""
    
    allow_reuse_address = True
    
    def __init__(self, port=8080):
        super().__init__(('localhost', port), StreamingHandler)
        self.camera_params = {'dist': 1.70, 'yaw': 180.0, 'pitch': 9.0, 'target': [0.6, 0.0, 0.4]}
        self.camera_params_file = None
        self.width = 640
        self.height = 480
        
    def set_camera_params_file(self, filepath):
        """Set path to camera parameters JSON file."""
        self.camera_params_file = filepath
        
    def update_camera_params(self):
        """Read camera parameters from file if available."""
        if self.camera_params_file and os.path.exists(self.camera_params_file):
            try:
                with open(self.camera_params_file, 'r') as f:
                    params = json.load(f)
                    self.camera_params['dist'] = params.get('dist', self.camera_params['dist'])
                    self.camera_params['yaw'] = params.get('yaw', self.camera_params['yaw'])
                    self.camera_params['pitch'] = params.get('pitch', self.camera_params['pitch'])
            except Exception:
                pass
    
    def get_frame(self):
        """Get current frame from PyBullet and return as JPEG bytes."""
        try:
            if not p.isConnected():
                return None
                
            # Update camera params
            self.update_camera_params()
            
            # Calculate camera position
            dist = self.camera_params['dist']
            yaw = math.radians(self.camera_params['yaw'])
            pitch = math.radians(self.camera_params['pitch'])
            target = self.camera_params['target']
            
            cam_x = target[0] + dist * math.cos(pitch) * math.sin(yaw)
            cam_y = target[1] + dist * math.cos(pitch) * math.cos(yaw)
            cam_z = target[2] + dist * math.sin(pitch)
            camera_pos = [cam_x, cam_y, cam_z]
            
            # Compute view and projection matrices
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=target,
                cameraUpVector=[0, 0, 1]
            )
            
            aspect = self.width / self.height
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=aspect,
                nearVal=0.01,
                farVal=100.0
            )
            
            # Get camera image
            img = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )
            
            # Convert to JPEG
            rgba = img[2]  # rgbPixels
            arr = np.array(rgba, dtype=np.uint8).reshape((self.height, self.width, 4))
            rgb = arr[:, :, :3]  # Drop alpha
            image = Image.fromarray(rgb)
            
            # Convert to bytes
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=85)
            return buf.getvalue()
            
        except Exception as e:
            print(f"[STREAM] Error getting frame: {e}")
            return None


def start_streaming_server(port=8080, camera_params_file=None):
    """Start the streaming server in a separate thread."""
    server = StreamingServer(port)
    if camera_params_file:
        server.set_camera_params_file(camera_params_file)
    
    def run_server():
        try:
            server.serve_forever()
        except Exception as e:
            print(f"[STREAM] Server error: {e}")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"[STREAM] Streaming server started on http://localhost:{port}/frame.jpg")
    return server


if __name__ == "__main__":
    # Test server
    import pybullet as p
    import pybullet_data
    
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    server = start_streaming_server(8080)
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240)
    except KeyboardInterrupt:
        server.shutdown()
        p.disconnect()

