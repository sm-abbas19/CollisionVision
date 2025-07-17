# CollisionVision/api_inference.py

import numpy as np
from collections import defaultdict, deque
from CollisionVision.detector.yolov8_wrapper import YOLOv8Wrapper
from CollisionVision.tracker.bytetrack_wrapper import ByteTrackWrapper
from CollisionVision.logic.collision import estimate_collision

TRAIL_LENGTH = 20  # Not used, but kept for reference

# Initialize detector and tracker once
detector = YOLOv8Wrapper()
tracker = ByteTrackWrapper(track_thresh=0.05, match_thresh=0.5, track_buffer=30, frame_rate=16)

prev_positions = {}
trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def process_image(frame):
    """
    Accepts a numpy image, returns detection, tracking, and collision estimation results.
    """
    # --- Detection ---
    detections = detector.detect(frame)
    detections = [
        (x1, y1, x2, y2, conf, int(class_id))
        for (x1, y1, x2, y2, conf, class_id) in detections
    ]
    detections = [det for det in detections if det[5] in [67, 39]]
    if detections:
        detections = np.array(detections, dtype=np.float32)
        detections[:, 5] = detections[:, 5].astype(np.int32)

    # --- Tracking ---
    tracked = tracker.update(
        detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]]
    )

    # --- Prepare tracked objects for collision logic ---
    tracked_objects = []
    current_ids = set()
    for obj in tracked:
        if obj['class_id'] not in [67, 39]:
            continue
        bbox = obj['bbox']
        track_id = obj['track_id']
        center = get_center(bbox)
        prev = prev_positions.get(track_id, center)
        jump_threshold = 100
        if np.linalg.norm(np.array(center) - np.array(prev)) > jump_threshold:
            prev = center
        tracked_objects.append({
            'track_id': track_id,
            'bbox': bbox,
            'pos_now': center,
            'pos_prev': prev,
            'class_id': obj['class_id']
        })
        prev_positions[track_id] = center
        current_ids.add(track_id)
    for tid in list(trails.keys()):
        if tid not in current_ids:
            del trails[tid]
            if tid in prev_positions:
                del prev_positions[tid]

    # --- Collision estimation (find closest pair) ---
    collision_pairs = []
    if len(tracked_objects) >= 2:
        pairs = []
        for i in range(len(tracked_objects)):
            for j in range(i + 1, len(tracked_objects)):
                obj1 = tracked_objects[i]
                obj2 = tracked_objects[j]
                dist = np.linalg.norm(np.array(obj1['pos_now']) - np.array(obj2['pos_now']))
                pairs.append((dist, obj1, obj2))
        pairs.sort(key=lambda x: x[0])
        if pairs:
            _, obj1, obj2 = pairs[0]
            prob, approaching, distance = estimate_collision(obj1, obj2)
            collision_pairs.append({
                "track_id_1": obj1['track_id'],
                "track_id_2": obj2['track_id'],
                "probability": prob,
                "approaching": approaching,
                "distance": distance
            })

    return {
        "tracked_objects": [
            {
                "track_id": obj['track_id'],
                "bbox": obj['bbox'],
                "pos_now": obj['pos_now'],
                "pos_prev": obj['pos_prev'],
                "class_id": obj['class_id']
            }
            for obj in tracked_objects
        ],
        "collision_pairs": collision_pairs
    }