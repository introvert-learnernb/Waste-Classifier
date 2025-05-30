import cv2
from PIL import Image
import matplotlib.pyplot as plt
from app.utils.constants import CLASS_NAMES, CLASS_GROUP_MAP
from app.utils.helpers import weighted_group_vote

def run_yolo_inference(model, image_paths, fallback_predict_func=None):
    """
    Run YOLO inference with fallback option.
    fallback_predict_func: function(image_path) → 'O' or 'R'
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for img_path in image_paths:
        print(f"\n[YOLO] Running inference on: {img_path}")

        results = model.predict(img_path, imgsz=640)

        all_boxes = []
        for r in results:
            all_boxes.extend(r.boxes)

        # Initial group prediction from YOLO
        group_pred, _ = weighted_group_vote(all_boxes, CLASS_NAMES, CLASS_GROUP_MAP)

        # Print details
        for box in all_boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cname = CLASS_NAMES[cls]
            group = CLASS_GROUP_MAP.get(cname, "Unknown")
            print(f"Class: {cname} → Group: {group}, Confidence: {conf:.2f}")

        if group_pred == "Unknown" and fallback_predict_func is not None:
            print("YOLO unsure. Falling back to DenseNet...")
            group_pred = fallback_predict_func(img_path)

        print(f"\nFinal Category: {group_pred}")

        # Show image, after final prediction
        img_with_boxes = results[0].plot()
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        plt.imshow(img_with_boxes)
        plt.title(f"Predicted → {group_pred}")
        plt.axis("off")
        plt.show()

        print("-" * 60)



def run_yolo_webcam(
    model, fallback_predict_func=None, threshold=0.5, cam_index=0, cooldown=3
):
    """
    Run YOLO on webcam with fallback classifier. Send prediction signal only after cooldown.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Unable to access webcam at index {cam_index}. Trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Still unable to access webcam. Exiting.")
            return

    print("Webcam running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, imgsz=640)
        img_with_boxes_rgb = results[0].plot()
        frame_with_boxes = cv2.cvtColor(img_with_boxes_rgb, cv2.COLOR_RGB2BGR)

        all_boxes = []
        for r in results:
            all_boxes.extend(r.boxes)

        print("\nYOLO Detections:")
        for box in all_boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cname = CLASS_NAMES[cls]
            group = CLASS_GROUP_MAP.get(cname, "Unknown")
            print(f"- {cname} → {group} ({conf:.2f})")

        final_group, _ = weighted_group_vote(
            all_boxes, CLASS_NAMES, CLASS_GROUP_MAP, threshold=threshold
        )

        used_fallback = False

        if final_group == "Unknown" and fallback_predict_func:
            print("YOLO uncertain → using DenseNet fallback...")
            pil_img = Image.fromarray(rgb_frame)
            final_group = fallback_predict_func(pil_img)
            used_fallback = True

        label = f"Prediction: {final_group}"
        if used_fallback:
            label += " (from DenseNet)"

        # Draw result
        cv2.putText(
            frame_with_boxes,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("YOLO + DenseNet (Webcam)", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
