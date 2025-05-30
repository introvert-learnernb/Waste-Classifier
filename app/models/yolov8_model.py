from ultralytics import YOLO


def load_yolov8_model(weights_path):
    """
    Loads the YOLOv8 model from weights.

    Args:
        weights_path (str): Path to the YOLOv8 weights file (.pt)

    Returns:
        model: Loaded YOLO model instance
    """
    model = YOLO(weights_path)
    return model
