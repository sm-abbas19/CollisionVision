import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
from collections import deque, defaultdict
from detector.yolov8_wrapper import YOLOv8Wrapper
from tracker.bytetrack_wrapper import ByteTrackWrapper
from logic.collision import estimate_collision
from utils.visualizer import visualize_frame

# --- Parameters ---
FRAME_SKIP = 1  # Run detection every N frames
TRAIL_LENGTH = 20  # Number of points in motion trail

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    detector = YOLOv8Wrapper()
    tracker = ByteTrackWrapper(track_thresh=0.001)
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
            detections = [det for det in detections if det[5] == 67]
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
        for obj in tracked:
            bbox = obj['bbox']
            track_id = obj['track_id']
            center = get_center(bbox)
            # Update trail
            trails[track_id].append(center)
            # Store previous position for velocity estimation
            prev = prev_positions.get(track_id, center)
            tracked_objects.append({
                'track_id': track_id,
                'bbox': bbox,
                'pos_now': center,
                'pos_prev': prev
            })
            prev_positions[track_id] = center

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
        out_frame = visualize_frame(frame, tracked_objects, trails, collision_pairs)
        cv2.imshow("Collision Estimator", out_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()