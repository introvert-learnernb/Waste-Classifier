import os
import random
from collections import defaultdict


def get_image_paths(folder_path, extensions=None, shuffle=True):
    """
    Returns a list of image file paths from the folder.
    Optionally filter by extensions and shuffle the list.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in extensions
    ]

    if shuffle:
        random.shuffle(files)

    return files


def weighted_group_vote(boxes, class_names, class_group_map, threshold=0.5):
    """
    Normalize total confidence by number of boxes per group.
    Returns the group with highest average confidence if above threshold.
    """

    group_confidence_sum = defaultdict(float)
    group_count = defaultdict(int)

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names[cls]
        group = class_group_map.get(class_name, "Unknown")

        group_confidence_sum[group] += conf
        group_count[group] += 1

    if not group_confidence_sum:
        return "Unknown", {}

    # Compute average (normalized) confidence per group
    group_avg_conf = {
        group: group_confidence_sum[group] / group_count[group]
        for group in group_confidence_sum
    }

    final_group = max(group_avg_conf, key=group_avg_conf.get)
    max_avg_conf = group_avg_conf[final_group]

    if max_avg_conf < threshold:
        return "Unknown", group_avg_conf

    return final_group, group_avg_conf
