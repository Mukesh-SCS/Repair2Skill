"""
HTTP server for streaming PyBullet simulation frames directly.
Uses a thread-safe buffer updated by the main simulation loop.
This design is necessary because PyBullet is NOT thread-safe.
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


class FrameBuffer:
    """Thread-safe buffer for sharing frames between main loop and HTTP server."""
    
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
    
    def set_frame(self, frame_bytes):
        """Set the current frame (called from main thread)."""
        with self._lock:
            self._frame = frame_bytes
    
    def get_frame(self):
        """Get the current frame (called from HTTP handler thread)."""
        with self._lock:
            return self._frame


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
                    frame = self.server.frame_buffer.get_frame()
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
            frame = self.server.frame_buffer.get_frame()
            if frame:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                # Return placeholder if no frame yet
                placeholder = self.server.get_placeholder_frame()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(placeholder)))
                self.end_headers()
                self.wfile.write(placeholder)
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


class StreamingServer(socketserver.TCPServer):
    """HTTP server that streams PyBullet frames from a shared buffer."""
    
    allow_reuse_address = True
    
    def __init__(self, port=8080, frame_buffer=None):
        super().__init__(('localhost', port), StreamingHandler)
        self.frame_buffer = frame_buffer or FrameBuffer()
        self.camera_params = {'dist': 1.70, 'yaw': 180.0, 'pitch': -30.0, 'target': [0.6, 0.0, 0.4]}
        self.camera_params_file = None
        self.width = 640
        self.height = 480
        self._placeholder_cache = None
        
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
    
    def get_placeholder_frame(self):
        """Generate a placeholder frame."""
        if self._placeholder_cache is None:
            placeholder = Image.new('RGB', (self.width, self.height), color=(60, 60, 80))
            buf = io.BytesIO()
            placeholder.save(buf, format='JPEG', quality=85)
            self._placeholder_cache = buf.getvalue()
        return self._placeholder_cache


def capture_frame(camera_params, width=640, height=480):
    """Capture a single frame from PyBullet.
    
    MUST be called from the main thread where PyBullet was initialized!
    
    Args:
        camera_params: dict with 'dist', 'yaw', 'pitch', 'target'
        width: frame width
        height: frame height
        
    Returns:
        JPEG bytes or None on error
    """
    try:
        if not p.isConnected():
            return None
            
        # Calculate camera position
        dist = camera_params['dist']
        yaw = math.radians(camera_params['yaw'])
        pitch = math.radians(camera_params['pitch'])
        target = camera_params.get('target', [0.6, 0.0, 0.4])
        
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
        
        aspect = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=aspect,
            nearVal=0.01,
            farVal=100.0
        )
        
        # Get camera image with proper lighting
        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            lightDirection=[0.4, 0.4, 1],
            lightColor=[1.0, 1.0, 1.0],
            lightDistance=2.0,
            lightAmbientCoeff=0.5,
            lightDiffuseCoeff=0.5,
            lightSpecularCoeff=0.3,
            renderer=p.ER_TINY_RENDERER
        )
        
        # Convert to JPEG
        rgba = img[2]  # rgbPixels
        arr = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
        rgb = arr[:, :, :3]  # Drop alpha
        image = Image.fromarray(rgb)
        
        # Convert to bytes
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
        
    except Exception as e:
        print(f"[STREAM] Error capturing frame: {e}")
        return None


def start_streaming_server(port=8080, camera_params_file=None):
    """Start the streaming server in a separate thread.
    
    Returns the server and frame buffer. The caller MUST call
    capture_frame() from the main simulation loop and update the buffer.
    
    Returns:
        StreamingServer instance
    """
    frame_buffer = FrameBuffer()
    server = StreamingServer(port, frame_buffer)
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
    import pybullet_data
    
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Create a test box
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(0, col, vis, [0.6, 0, 0.4])
    
    server = start_streaming_server(8080)
    camera_params = {'dist': 1.7, 'yaw': 180, 'pitch': -30, 'target': [0.6, 0, 0.4]}
    
    try:
        frame_count = 0
        while True:
            p.stepSimulation()
            
            # Capture frame from main thread and update buffer
            if frame_count % 2 == 0:  # Capture at ~30 FPS
                frame = capture_frame(camera_params)
                if frame:
                    server.frame_buffer.set_frame(frame)
            
            frame_count += 1
            time.sleep(1.0 / 60)
    except KeyboardInterrupt:
        server.shutdown()
        p.disconnect()

