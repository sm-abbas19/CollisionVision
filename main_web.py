import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
from collections import deque, defaultdict
from CollisionVision.detector.yolov8_wrapper import YOLOv8Wrapper
from CollisionVision.tracker.bytetrack_wrapper import ByteTrackWrapper
from CollisionVision.logic.collision import estimate_collision
from CollisionVision.utils.visualizer import visualize_frame
import time
import psutil

# -------------------- ADD THESE IMPORTS --------------------
from CollisionVision.database.database import DatabaseManager

# --- Parameters ---
FRAME_SKIP = 1  # Run detection every N frames
TRAIL_LENGTH = 20  # Number of points in motion trail

# -------------------- EVENT LOGGER CLASS --------------------
class EventLogger:
    def __init__(self):
        self.db = DatabaseManager()

    def classify_event(self, probability, distance):
        if probability > 0.9:
            return "collision", "high"
        elif probability > 0.5:
            return "near_miss", "medium"
        elif distance < 50:  # pixels
            return "warning", "low"
        return None, None

    def log_event(self, obj1, obj2, probability, distance, frame_idx):
        event_type, severity = self.classify_event(float(probability), float(distance))
        if event_type:
            event_data = {
                'event_type': event_type,
                'probability': float(probability),
                'distance': float(distance),
                'object1_id': int(obj1['track_id']),
                'object2_id': int(obj2['track_id']),
                'object1_class': int(obj1['class_id']),
                'object2_class': int(obj2['class_id']),
                'frame_number': int(frame_idx),
                'severity': severity
            }
            self.db.log_collision_event(event_data)

    def log_system_metrics(self, fps, total_objects, active_tracks,cpu_usage, mem_usage):
        metrics_data = {
            'fps': fps,
            'total_objects_detected': total_objects,
            'active_tracks': active_tracks,
            'cpu_usage': cpu_usage,
            'memory_usage': mem_usage

        }
        self.db.log_system_metrics(metrics_data)

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def main():
    prev_time = time.time()
    fps = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    # Get the actual camera FPS
    frame_rate = 16
    print(f"Camera reports FPS: {frame_rate}")

    # If the value is 0 or very low, default to 30
    if frame_rate < 1:
        frame_rate = 30

    detector = YOLOv8Wrapper()
    tracker = ByteTrackWrapper(track_thresh=0.05, match_thresh=0.5,track_buffer=30)
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    prev_positions = {}
    logger = EventLogger()  # ---------------- Add event logger -----------------

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_info = {
            'height': frame.shape[0],
            'width': frame.shape[1],
            'img': frame
        }

        detections = []
        if frame_idx % FRAME_SKIP == 0:
            detections = detector.detect(frame)
            # Ensure class_id is int in each tuple
            detections = [
                (x1, y1, x2, y2, conf, int(class_id))
                for (x1, y1, x2, y2, conf, class_id) in detections
            ]
            detections = [det for det in detections if det[5] in [67, 39]]
            if detections:
                detections = np.array(detections, dtype=np.float32)
                detections[:, 5] = detections[:, 5].astype(np.int32)
            # print("Detections to tracker:", detections)

        tracked = tracker.update(
            detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]]
        )

        tracked_objects = []
        current_ids = set()
        for obj in tracked:
            if obj['class_id'] not in [67, 39]:
                continue
            bbox = obj['bbox']
            track_id = obj['track_id']
            center = get_center(bbox)
            prev = prev_positions.get(track_id, center)
            jump_threshold = 100  # pixels
            if np.linalg.norm(np.array(center) - np.array(prev)) > jump_threshold:
                trails[track_id].clear()
                prev = center
            trails[track_id].append(center)

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

        # ------------------- LOG COLLISION EVENTS ------------------
        # Find two closest objects for collision estimation
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
                distance, obj1, obj2 = pairs[0]
                prob, approaching, _distance = estimate_collision(obj1, obj2)
                collision_pairs.append((obj1, obj2, prob))
                # Log the event
                logger.log_event(obj1, obj2, prob, distance, frame_idx)

        # ------------- LOG SYSTEM METRICS EVERY N FRAMES -------------
        if frame_idx % 100 == 0:  # Log every 100 frames
            cpu = psutil.cpu_percent(interval=None)  # get CPU usage %
            mem = psutil.virtual_memory().percent
            logger.log_system_metrics(fps, len(tracked_objects), len(current_ids), cpu, mem)

        # Visualize
        out_frame = visualize_frame(frame, tracked_objects, trails, collision_pairs, frame_idx)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(out_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collision Estimator", out_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()