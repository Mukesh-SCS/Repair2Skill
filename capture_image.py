from picamera2 import Picamera2
from PIL import Image
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

picam2.start()
time.sleep(2)  # Camera warm-up

image = picam2.capture_array()
img = Image.fromarray(image)
img.save("chair.jpg")
print("✅ Chair image saved as 'chair.jpg'")
picam2.stop()
