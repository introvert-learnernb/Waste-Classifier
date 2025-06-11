import subprocess
import sys
import os
from dotenv import load_dotenv
from app.services.model_loader import get_models
from app.inference.yolov8_inference import run_yolo_webcam
from app.inference.densenet201_inference import densenet_inference

# Load variables from .env file
load_dotenv()

# Load models
_, densenet_model, best_model = get_models()


def runwebcam():
    run_yolo_webcam(
        best_model,
        fallback_predict_func=lambda img: densenet_inference(densenet_model, img),
    )


def runserver():
    """Run FastAPI app with uvicorn using env vars for host and port."""

    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = os.getenv("FASTAPI_PORT", "8000")

    print(f"Starting FastAPI server on {host}:{port}...")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                host,
                "--port",
                port,
                "--reload",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Failed to start uvicorn server.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python manage.py runserver")
        sys.exit(1)

    command = sys.argv[1]

    if command == "runserver":
        runserver()
    if command == "runwebcam":
        runwebcam()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python manage.py runserver")
        sys.exit(1)


if __name__ == "__main__":
    main()
