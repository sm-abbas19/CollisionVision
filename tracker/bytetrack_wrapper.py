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
        #orig_dets = dets.copy() if dets.shape[1] == 6 else None
        if dets.shape[1] == 5:
            dets = np.hstack([dets, np.zeros((dets.shape[0], 1))])
            #dets = dets[:, :5]
        online_targets = self.tracker.update(
            dets, img_info, img_size
        )
        results = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            results.append({
                'track_id': int(t.track_id),
                'bbox': (x1, y1, x2, y2),
                'score': float(t.score),
                'class_id': int(t.class_id)
            })
        return results