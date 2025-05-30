# Waste Classification API

This FastAPI application identifies waste using **YOLOv8s** (object detection) and **DenseNet201** (image classification). It classifies waste into two groups: **Recyclable (R)** or **Organic (O)**.

---

## Quickstart

1. **Get the code:**

    ```bash
    git clone https://github.com/ManishJoc14/Waste-Classifier.git
    cd Waste-Classifier
    ```

2. **Set up environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Or `source venv/bin/activate` on Unix
    pip install -r requirements.txt
    ```

    Rename `.env.example` to `.env` and adjust your values as needed.

3. **Run the server:**

    ```bash
    python manage.py runserver
    ```

---

## API Endpoint

- **Image Upload:**
  - **POST** `/api/v1/upload`  
    Accepts:
    - A standard image file via `multipart/form-data` (PNG, JPG, etc.)
    - A raw RGB565 byte stream (`application/octet-stream`)
  - **Response:**
    ```json
    200 OK
    { "prediction": "O", 'confidence': 0.9030410945415497 }
    ```

    Where `'O'` = Organic and `'R'` = Recyclable

---

## Details

- **Models:**  
  - [YOLOv8s](https://github.com/ultralytics/ultralytics) for object detection  
  - [DenseNet201](https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html) for classification fallback
- **Dataset:**  
  Based on the [TACO Dataset (YOLO format)](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format/code)  
  Official: [tacodataset.org](http://tacodataset.org/explore)
- **Saved Predictions:**  
  Processed output images (with predictions) are stored in `app/data/saved_images/` when `DEBUG=True` in `.env`
- **Cooldown:**  
  Inference requests are rate-limited to prevent excessive load.

---

## Testing

Run test uploads to the API:

```bash
python -m app.tests.test_upload
```

You can toggle between:

- **Normal image upload** (JPG/PNG): `use_rgb565=False` (default)
- **Raw RGB565 byte upload**: `use_rgb565=True` (for Arduino-like testing)

Edit this line in `test_upload.py`:

```python
test_image_upload(use_rgb565=False)  # Set to True to test RGB565 mode
```

---
