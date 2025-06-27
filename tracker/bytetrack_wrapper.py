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
        """
        Args:
            detections: List of (x1, y1, x2, y2, conf, class_id)
            img_info: list [height, width] of the frame
            img_size: list [height, width] of the frame

        Returns:
            List of tracked objects:
            [{
                'track_id': int,
                'bbox': (x1, y1, x2, y2),
                'score': float,
                'class_id': int
            }, ...]
        """
        dets = detections

        print("dets passed to BYTETracker:", dets) 

        online_targets = self.tracker.update(
            dets,
            img_info,   # img_info is now [height, width]
            img_size    # img_size is also [height, width]
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
                'class_id': int(getattr(t, 'cls', -1))
            })
        return results