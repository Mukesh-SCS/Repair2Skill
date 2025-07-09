# ==================== scripts/capture_image.py ====================
import cv2
import os
from datetime import datetime
import numpy as np

def capture_from_camera(save_dir="./data/user_images/"):
    """Capture image from Pi Camera or webcam"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Try Pi Camera first, fallback to webcam
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        
        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(save_dir, f"captured_{timestamp}.jpg")
        
        # Give camera time to adjust
        import time
        time.sleep(2)
        
        # Capture and save
        picam2.capture_file(image_path)
        picam2.stop()
        
        print(f"Image captured successfully: {image_path}")
        return image_path
        
    except ImportError:
        print("Pi Camera not available, using webcam...")
        
        # Fallback to webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(save_dir, f"captured_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        
        cap.release()
        print(f"Image captured successfully: {image_path}")
        return image_path
