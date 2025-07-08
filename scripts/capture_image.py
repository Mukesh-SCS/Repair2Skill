from picamera2 import Picamera2
from PIL import Image
import time, os

def capture_from_camera():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(2)

    image = picam2.capture_array()
    img = Image.fromarray(image)
    os.makedirs("./data/user_images", exist_ok=True)
    image_path = "./data/user_images/captured_image.jpg"
    img.save(image_path)
    picam2.stop()

    return image_path
