from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from collections import deque, defaultdict
from CollisionVision.detector.yolov8_wrapper import YOLOv8Wrapper
from CollisionVision.tracker.bytetrack_wrapper import ByteTrackWrapper
from CollisionVision.logic.collision import estimate_collision

app = FastAPI()

TRAIL_LENGTH = 20
detector = YOLOv8Wrapper()
tracker = ByteTrackWrapper(track_thresh=0.05, match_thresh=0.5, track_buffer=30, frame_rate=30)
trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
prev_positions = {}

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_info = {
        'height': frame.shape[0],
        'width': frame.shape[1],
        'img': frame
    }

    detections = detector.detect(frame)
    detections = [
        (x1, y1, x2, y2, conf, int(class_id))
        for (x1, y1, x2, y2, conf, class_id) in detections
    ]
    detections = [det for det in detections if det[5] in [67, 39]]
    if detections:
        detections = np.array(detections, dtype=np.float32)
        detections[:, 5] = detections[:, 5].astype(np.int32)

    tracked = tracker.update(detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])

    tracked_objects = []
    current_ids = set()
    for obj in tracked:
        if obj['class_id'] not in [67, 39]:
            continue
        bbox = obj['bbox']
        track_id = obj['track_id']
        center = get_center(bbox)
        prev = prev_positions.get(track_id, center)
        velocity = (center[0] - prev[0], center[1] - prev[1])
        jump_threshold = 100
        if np.linalg.norm(np.array(center) - np.array(prev)) > jump_threshold:
            trails[track_id].clear()
            prev = center
            velocity = (0, 0)
        trails[track_id].append(center)
        tracked_objects.append({
            'track_id': track_id,
            'bbox': [float(b) for b in bbox],
            'center': [float(center[0]), float(center[1])],
            'velocity': [float(velocity[0]), float(velocity[1])],
            'class_id': int(obj['class_id'])
        })
        prev_positions[track_id] = center
        current_ids.add(track_id)
    for tid in list(trails.keys()):
        if tid not in current_ids:
            del trails[tid]
            if tid in prev_positions:
                del prev_positions[tid]

    collision_pairs = []
    if len(tracked_objects) >= 2:
        pairs = []
        for i in range(len(tracked_objects)):
            for j in range(i + 1, len(tracked_objects)):
                obj1 = tracked_objects[i]
                obj2 = tracked_objects[j]
                dist = np.linalg.norm(np.array(obj1['center']) - np.array(obj2['center']))
                pairs.append((dist, obj1, obj2))
        pairs.sort(key=lambda x: x[0])
        if pairs:
            _, obj1, obj2 = pairs[0]
            obj1_for_collision = {
                'pos_now': obj1['center'],
                'pos_prev': [obj1['center'][0] - obj1['velocity'][0], obj1['center'][1] - obj1['velocity'][1]]
            }
            obj2_for_collision = {
                'pos_now': obj2['center'],
                'pos_prev': [obj2['center'][0] - obj2['velocity'][0], obj2['center'][1] - obj2['velocity'][1]]
    }
            prob, approaching, distance = estimate_collision(obj1_for_collision, obj2_for_collision)
            collision_pairs.append({
                'track_id_1': obj1['track_id'],
                'track_id_2': obj2['track_id'],
                'collision_probability': float(prob),
                'distance': float(distance),
                'approaching': bool(approaching)
            })

    return JSONResponse({
        "tracked_objects": tracked_objects,
        "collision_pairs": collision_pairs
    })