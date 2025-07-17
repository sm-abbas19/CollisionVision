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

# --- Parameters ---
FRAME_SKIP = 1  # Run detection every N frames
TRAIL_LENGTH = 20  # Number of points in motion trail

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
            print("Detections to tracker:", detections)
            #print("Detections to tracker:", detections)

        #for det in detections:
             #x1, y1, x2, y2, conf, class_id = det
             #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
             #cv2.putText(frame, f"{int(class_id)} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Tracking
        tracked = tracker.update(detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
        print("Tracked objects:", tracked)

        # Prepare tracked objects for visualization and collision logic
        tracked_objects = []
        current_ids = set()
        for obj in tracked:
            if obj['class_id'] not in [67, 39]:
                continue
            bbox = obj['bbox']
            track_id = obj['track_id']
            center = get_center(bbox)
            # Update trail
            #trails[track_id].append(center)
            # Store previous position for velocity estimation
            prev = prev_positions.get(track_id, center)
            # --- Large jump filter ---
            jump_threshold = 100  # pixels, adjust as needed
            if np.linalg.norm(np.array(center) - np.array(prev)) > jump_threshold:
                trails[track_id].clear()
                prev = center
            # Update trail
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
        # --- Remove trails for tracks not present in this frame ---
        for tid in list(trails.keys()):
            if tid not in current_ids:
                del trails[tid]
                if tid in prev_positions:
                    del prev_positions[tid]

        # Find two closest objects for collision estimation
        collision_pairs = []
        if len(tracked_objects) >= 2:
            # Compute pairwise distances
            pairs = []
            for i in range(len(tracked_objects)):
                for j in range(i + 1, len(tracked_objects)):
                    obj1 = tracked_objects[i]
                    obj2 = tracked_objects[j]
                    dist = np.linalg.norm(np.array(obj1['pos_now']) - np.array(obj2['pos_now']))
                    pairs.append((dist, obj1, obj2))
            # Sort by distance and pick the closest pair
            pairs.sort(key=lambda x: x[0])
            if pairs:
                _, obj1, obj2 = pairs[0]
                prob, approaching, distance = estimate_collision(obj1, obj2)
                                                            
                collision_pairs.append((obj1, obj2, prob))

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