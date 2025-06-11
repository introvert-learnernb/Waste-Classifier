from ultralytics import YOLO


def load_best_model(weights_path):
    """
    Loads the Best model from weights.

    Args:
        weights_path (str): Path to the Best weights file (.pt)

    Returns:
        model: Loaded YOLO BEST model instance
    """
    model = YOLO(weights_path)
    return model
