import torch
from PIL import Image
from app.utils.constants import DEVICE
from app.utils.transforms import test_transform


def densenet_inference(model, image_input):
    """
    Returns 'O' or 'R' from DenseNet prediction.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    img_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
    densenet_class_names = ["O", "R"]

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        pred_class = densenet_class_names[pred_class_idx]

    print(f"[DenseNet] Prediction: {pred_class}")
    return pred_class
