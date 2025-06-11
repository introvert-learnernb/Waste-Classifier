import io
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi import Request, APIRouter, File, UploadFile
from app.services.model_loader import get_models
from app.inference.arduino_inference import run_arduino_inference
from app.inference.densenet201_inference import densenet_inference
from app.services.arduino_helpers import (
    validate_raw_bytes,
    rgb565_bytes_to_rgb_image,
    optionally_save_image,
)

# Load models
yolo_model, densenet_model = get_models()


router = APIRouter()


@router.post("")
async def upload(request: Request, file: UploadFile = File(None)):
    try:
        if file:
            # Standard image upload (JPG/PNG)
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        else:
            # Raw byte stream (e.g., RGB565 from Arduino)
            raw_data = await request.body()
            validate_raw_bytes(raw_data)
            image = rgb565_bytes_to_rgb_image(raw_data)

        # Run YOLO + fallback DenseNet inference
        prediction = run_arduino_inference(
            yolo_model,
            image,
            fallback_predict_func=lambda img: densenet_inference(densenet_model, img),
        )

        optionally_save_image(image, prediction)

        final_group, final_confidence, _ = prediction
        return {
            "prediction": final_group,
            "confidence": final_confidence,
        }

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
