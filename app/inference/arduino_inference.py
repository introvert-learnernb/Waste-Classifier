
from app.utils.helpers import weighted_group_vote
from app.utils.constants import CLASS_NAMES, CLASS_GROUP_MAP
from app.services.arduino_helpers import send_to_arduino

# Initialize cooldown tracking


def run_arduino_inference(model, image, fallback_predict_func=None, threshold=0.5):
    

    results = model.predict(image, imgsz=640)
    all_boxes = []
    for r in results:
        all_boxes.extend(r.boxes)

    final_group, group_conf = weighted_group_vote(
        all_boxes, CLASS_NAMES, CLASS_GROUP_MAP, threshold=threshold
    )

    if final_group == "Unknown" and fallback_predict_func:
        final_group, confidence = fallback_predict_func(image)
        final_confidence = confidence
    else:
        final_confidence = group_conf.get(final_group, None)

    if final_group != "Unknown":
        send_to_arduino(final_group)


    return final_group, final_confidence, all_boxes
