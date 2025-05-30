import torch
import torch.nn as nn
from torchvision import models
from app.utils.constants import DEVICE


def get_densenet201_model(num_classes=2, pretrained=True):
    model = models.densenet201(pretrained=pretrained)

    # Freeze pre-trained weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier layer to fit num_classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return model.to(DEVICE)


def load_densenet201_model(model_path, num_classes=2):
    """
    Loads the densenet-201 model from weights.

    Args:
        weights_path (str): Path to the densenet-201 weights file (.pth)

    Returns:
        model: Loaded densenet model instance
    """
    model = get_densenet201_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model
