import subprocess
import sys
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


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
    else:
        print(f"Unknown command: {command}")
        print("Usage: python manage.py runserver")
        sys.exit(1)


if __name__ == "__main__":
    main()
