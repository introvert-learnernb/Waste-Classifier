import os
import time
import serial
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.utils.constants import (
    HEIGHT,
    WIDTH,
    DEBUG,
    MAX_SAVE,
    EXPECTED_SIZE,
    INFERENCE_IMAGE_SAVE_DIR,
    SAVE_COUNT,
    CLASS_NAMES,
)


arduino = None


def send_to_arduino(pred_group: str):
    global arduino
    if pred_group not in {"O", "R"}:
        print(f"[Serial] Invalid group '{pred_group}' — not sending.")
        return

    try:
        if arduino is None or not arduino.is_open:
            arduino = serial.Serial("COM7", 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to be ready

        command = b"L\n" if pred_group == "O" else b"R\n"
        arduino.write(command)
        print(f"[Serial] Sent: {command.strip().decode()}")

    except serial.SerialException as e:
        print(f"[Error] Serial connection issue: {e}")

    except Exception as e:
        print(f"[Error] Unexpected error: {e}")


def validate_raw_bytes(raw_bytes: bytes):
    """Validate that raw bytes length matches expected size."""
    if len(raw_bytes) != EXPECTED_SIZE:
        raise ValueError(
            f"Invalid image size: {len(raw_bytes)} bytes (expected {EXPECTED_SIZE})"
        )


def rgb565_bytes_to_rgb_image(raw_bytes: bytes):
    """Convert raw RGB565 bytes (big endian) to PIL RGB image."""
    arr = np.frombuffer(
        raw_bytes, dtype=np.uint16
    ).byteswap()  # Correct endian, shape (width*height,)

    # Extract RGB components
    r = (arr >> 11) & 0x1F
    g = (arr >> 5) & 0x3F
    b = arr & 0x1F

    # Scale up to 8-bit
    r = (r * 255 // 31).astype(np.uint8)
    g = (g * 255 // 63).astype(np.uint8)
    b = (b * 255 // 31).astype(np.uint8)

    # Stack and reshape into image array
    rgb = np.stack([r, g, b], axis=1).reshape((HEIGHT, WIDTH, 3))

    return Image.fromarray(rgb, "RGB")


def rgb_image_to_rgb565_bytes(image):
    """Convert RGB PIL image (HxW) to RGB565 byte stream (big endian)."""
    arr = np.array(image, dtype=np.uint8)

    # Extract channels and reduce bit depth properly:
    r = (arr[:, :, 0] >> 3).astype(np.uint16)  # 5 bits
    g = (arr[:, :, 1] >> 2).astype(np.uint16)  # 6 bits
    b = (arr[:, :, 2] >> 3).astype(np.uint16)  # 5 bits

    # Compose RGB565 pixel
    rgb565 = (r << 11) | (g << 5) | b  # 16-bit values

    # Convert to bytes (big endian): swap bytes because numpy default is little endian on x86
    return rgb565.byteswap().tobytes()


def optionally_save_image(image: Image.Image, prediction: tuple):
    """
    Save prediction image with timestamp and overwrite latest.png.
    Draw bounding boxes and predicted label on image.

    Args:
        image: PIL Image
        prediction: tuple of (final_group: str, final_confidence: float,  all_boxes: list)
    """
    global SAVE_COUNT

    if not DEBUG:
        return

    final_group, _, all_boxes = prediction

    # Copy image to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try loading a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Draw all bounding boxes with labels
    for box in all_boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        cls = int(box.cls[0])  # class index
        conf = box.conf[0]  # confidence score

        label = f"{CLASS_NAMES[cls]} {conf:.2f}"

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Draw label background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
        draw.text((xmin, ymin - text_height), label, fill="white", font=font)

    # Draw predicted group label at top-left corner
    group_label = f"Predicted Group: {final_group}"
    draw.text((10, 10), group_label, fill="blue", font=font)

    # Ensure save directory exists
    os.makedirs(INFERENCE_IMAGE_SAVE_DIR, exist_ok=True)

    if SAVE_COUNT >= MAX_SAVE:
        # Delete all files except latest.png
        for filename in os.listdir(INFERENCE_IMAGE_SAVE_DIR):
            if filename != "latest.png":
                try:
                    os.remove(os.path.join(INFERENCE_IMAGE_SAVE_DIR, filename))
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
        SAVE_COUNT = 0  # reset counter after cleanup

    if SAVE_COUNT < MAX_SAVE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{final_group}_{timestamp}.png"
        path = os.path.join(INFERENCE_IMAGE_SAVE_DIR, filename)
        img_draw.save(path)
        SAVE_COUNT += 1

    # Always overwrite latest.png in actual RGB
    latest_path = os.path.join(INFERENCE_IMAGE_SAVE_DIR, "latest.png")
    img_draw.save(latest_path)
