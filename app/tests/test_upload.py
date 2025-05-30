import os, io
import requests
from PIL import Image
from dotenv import load_dotenv
from app.utils.helpers import get_image_paths
from app.utils.constants import HEIGHT, WIDTH, TEST_IMAGE_DIR
from app.services.arduino_helpers import rgb_image_to_rgb565_bytes

load_dotenv()

host = os.getenv("FASTAPI_HOST", "127.0.0.1")
port = os.getenv("FASTAPI_PORT", "8000")
API_URL = f"http://{host}:{port}/api/v1/upload"


def test_image_upload(use_rgb565=False):
    image_paths = get_image_paths(TEST_IMAGE_DIR, shuffle=True)
    if not image_paths:
        print("No images found.")
        return

    path = image_paths[0]
    print(f"Using image: {path}")

    image = Image.open(path).convert("RGB")

    if use_rgb565:
        # For raw RGB565 byte stream
        image = image.resize((WIDTH, HEIGHT))
        payload = rgb_image_to_rgb565_bytes(image)
        headers = {"Content-Type": "application/octet-stream"}
        print("Sending raw RGB565 bytes...")
        response = requests.post(API_URL, data=payload, headers=headers)

    else:
        # For normal image file (e.g., JPEG/PNG)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        files = {"file": ("test.jpg", buf, "image/jpeg")}
        print("Sending image as multipart/form-data...")
        response = requests.post(API_URL, files=files)

    print("Response:", response.status_code, response.json())


if __name__ == "__main__":
    # Change this to True to test raw RGB565 mode
    # Change this to False to test multipart/form-data
    test_image_upload(use_rgb565=False)
