import numpy as np
from types import SimpleNamespace

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    raise ImportError("Please install ByteTrack and its dependencies.")

class ByteTrackWrapper:
    def __init__(self, track_thresh=0.3, match_thresh=0.8, track_buffer=30, frame_rate=30):
        # Create an args namespace as expected by BYTETracker
        args = SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            mot20=False
        )
        self.tracker = BYTETracker(args, frame_rate)

    def update(self, detections, img_info, img_size):
        if len(detections) == 0:
            return []
        dets = np.asarray(detections)
        orig_dets = dets.copy() if dets.shape[1] == 6 else None
        if dets.shape[1] == 6:
            dets = dets[:, :5]
        online_targets = self.tracker.update(
            dets, img_info, img_size
        )
        results = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            class_id = -1
            if orig_dets is not None:
                # Find detection with highest IoU
                best_iou = 0
                best_idx = -1
                for i, det in enumerate(orig_dets):
                    dx1, dy1, dx2, dy2 = det[:4]
                    inter_x1 = max(x1, dx1)
                    inter_y1 = max(y1, dy1)
                    inter_x2 = min(x2, dx2)
                    inter_y2 = min(y2, dy2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (dx2 - dx1) * (dy2 - dy1)
                    union_area = area1 + area2 - inter_area + 1e-6
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx >= 0:
                    class_id = int(orig_dets[best_idx][5])
            results.append({
                'track_id': int(t.track_id),
                'bbox': (x1, y1, x2, y2),
                'score': float(t.score),
                'class_id': class_id
            })
        return results