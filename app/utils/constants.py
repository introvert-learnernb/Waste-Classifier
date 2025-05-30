import torch

# Model input image size
IMG_SIZE = 224

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = [
    "Aluminium foil",
    "Bottle cap",
    "Bottle",
    "Broken glass",
    "Can",
    "Carton",
    "Cigarette",
    "Cup",
    "Lid",
    "Other litter",
    "Other plastic",
    "Paper",
    "Plastic bag - wrapper",
    "Plastic container",
    "Pop tab",
    "Straw",
    "Styrofoam piece",
    "Unlabeled litter",
]

# Grouped classes
ORGANIC_CLASSES = [
    "Carton",
    "Cigarette",
    "Cup",
    "Lid",
    "Other litter",
    "Paper",
    "Straw",
    "Unlabeled litter",
]

RECYCLABLE_CLASSES = [
    "Aluminium foil",
    "Bottle cap",
    "Bottle",
    "Broken glass",
    "Can",
    "Other plastic",
    "Plastic bag - wrapper",
    "Plastic container",
    "Pop tab",
    "Styrofoam piece",
]

# Class-to-group map (e.g., 'O' = Organic, 'R' = Recyclable)
CLASS_GROUP_MAP = {cls: "O" for cls in ORGANIC_CLASSES}
CLASS_GROUP_MAP.update({cls: "R" for cls in RECYCLABLE_CLASSES})

DENSENET_MODEL_PATH = "app/weights/densenet201_model.pth"
YOLO_MODEL_PATH = "app/weights/yolov8_model.pt"


# Image config
WIDTH, HEIGHT = 320, 240
EXPECTED_SIZE = WIDTH * HEIGHT * 2
INFERENCE_IMAGE_SAVE_DIR = "app/data/saved_images"
TEST_IMAGE_DIR = "app/data/test_images"

DEBUG = True
SAVE_COUNT = 0
MAX_SAVE = 10

BASE_PREFIX = "/api/v1"
