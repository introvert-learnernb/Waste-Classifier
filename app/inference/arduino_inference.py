import time
from app.utils.helpers import weighted_group_vote
from app.utils.constants import CLASS_NAMES, CLASS_GROUP_MAP
from app.services.arduino_helpers import send_to_arduino

# Initialize cooldown tracking
last_prediction_time = 0
last_group_sent = None
COOLDOWN = 3  # seconds


def run_arduino_inference(model, image, fallback_predict_func=None, threshold=0.5):
    global last_prediction_time, last_group_sent
    results = model.predict(image, imgsz=640)
    all_boxes = []
    for r in results:
        all_boxes.extend(r.boxes)

    final_group, group_conf = weighted_group_vote(
        all_boxes, CLASS_NAMES, CLASS_GROUP_MAP, threshold=threshold
    )

    if final_group == "Unknown" and fallback_predict_func:
        final_group = fallback_predict_func(image)
        final_confidence = None  # No confidence from fallback
    else:
        final_confidence = group_conf.get(final_group, None)

    now = time.time()
    if final_group != "Unknown" and final_group != last_group_sent:
        if now - last_prediction_time > COOLDOWN:
            send_to_arduino(final_group)
            last_prediction_time = now
            last_group_sent = final_group

    return final_group, final_confidence, all_boxes
