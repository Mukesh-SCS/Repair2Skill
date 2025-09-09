"""
================================================================================
DESCRIPTION:
    Capture image from Raspberry Pi Camera if available, else fallback to webcam.

USAGE:
    from scripts.capture_image import capture_from_camera
    path = capture_from_camera()

OUTPUTS:
    ./data/user_images/captured_<timestamp>.jpg

ARGUMENTS:
    save_dir: output directory for captured images
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os
from datetime import datetime
import cv2


def capture_from_camera(save_dir="./data/user_images/"):
    os.makedirs(save_dir, exist_ok=True)
    try:
        from picamera2 import Picamera2  # type: ignore
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"captured_{ts}.jpg")
        import time
        time.sleep(2)
        picam2.capture_file(path)
        picam2.stop()
        print(f"Image captured: {path}")
        return path
    except ImportError:
        print("Pi Camera not available. Using webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Failed to capture image")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"captured_{ts}.jpg")
        cv2.imwrite(path, frame)
        cap.release()
        print(f"Image captured: {path}")
        return path
