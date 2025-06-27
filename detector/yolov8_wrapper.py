import numpy as np
from ultralytics import YOLO

class YOLOv8Wrapper:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        """
        Run YOLOv8n detection on a single frame.

        Args:
            frame (np.ndarray): Input image (BGR, uint8).

        Returns:
            List of detections: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        results = self.model(frame, device=self.device, verbose=False)
        
      
        detections = []
        

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                detections.append((x1, y1, x2, y2, conf, class_id))
        return detections