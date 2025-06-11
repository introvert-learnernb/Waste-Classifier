from app.models.yolov8_model import load_yolov8_model
from app.models.densenet201_model import load_densenet201_model
from app.models.best_model import load_best_model
from app.utils.constants import YOLO_MODEL_PATH, DENSENET_MODEL_PATH, BEST_MODEL_PATH


def get_models():
    yolo = load_yolov8_model(YOLO_MODEL_PATH)
    densenet = load_densenet201_model(DENSENET_MODEL_PATH)
    best = load_best_model(BEST_MODEL_PATH)
    return yolo, densenet, best
