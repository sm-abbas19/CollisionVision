import cv2
import numpy as np

def draw_bbox(frame, bbox, obj_id, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"ID {obj_id}"
    if label is not None:
        text += f" {label}"
    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_trail(frame, trail, color=(255, 0, 0)):
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        pt1 = tuple(map(int, trail[i - 1]))
        pt2 = tuple(map(int, trail[i]))
        cv2.line(frame, pt1, pt2, color, 2)

def draw_collision_prob(frame, obj1, obj2, prob, color=(0, 0, 255)):
    # Draw a line between the two objects' current positions
    pt1 = tuple(map(int, obj1['pos_now']))
    pt2 = tuple(map(int, obj2['pos_now']))
    cv2.line(frame, pt1, pt2, color, 2)
    # Place the probability text at the midpoint
    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    text = f"Collision: {prob:.2f}"
    cv2.putText(frame, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def visualize_frame(frame, tracked_objects, trails, collision_pairs):
    """
    Args:
        frame: np.ndarray (BGR)
        tracked_objects: list of dicts with keys 'track_id', 'bbox', 'pos_now', 'class_id'
        trails: dict {track_id: deque of (x, y)}
        collision_pairs: list of (obj1, obj2, prob)
    """
    # Draw bounding boxes and IDs
    for obj in tracked_objects:
        print("DEBUG class_id:", obj.get('class_id', None), type(obj.get('class_id', None)))
        # Assign color and label based on class_id
        if (obj.get('class_id', None)) == 67:
            color = (128, 0, 128)      # purple for phone
            label = "Phone"
        elif (obj.get('class_id', None)) == 39:
            color = (255, 0, 255)    # Magenta for bottle
            label = "Bottle"
        else:
            color = (200, 200, 200)  # Gray for others
            label = str(obj.get('class_id', ''))
        draw_bbox(frame, obj['bbox'], obj['track_id'], color, label)
        # Draw trail if available
        if obj['track_id'] in trails:
            draw_trail(frame, trails[obj['track_id']], color=color)
    # Draw collision probabilities
    for obj1, obj2, prob in collision_pairs:
        if prob > 0:
            draw_collision_prob(frame, obj1, obj2, prob)
    return frame