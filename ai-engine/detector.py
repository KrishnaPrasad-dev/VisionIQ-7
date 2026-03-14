import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np

class PersonDetector:
    def __init__(self):
        print('Loading YOLO11s model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        self.model = YOLO("yolo11s.pt")
        self.model.to(self.device)

        # ByteTrack for persistent IDs
        self.tracker = sv.ByteTrack()

        # Warmup — use float32 for warmup to avoid dtype conflict
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, classes=[0], conf=0.5, verbose=False)
        print(f'YOLO11s loaded and warmed up on {self.device}!')

    def detect(self, frame: np.ndarray) -> sv.Detections:
        if frame is None or frame.size == 0:
            return sv.Detections.empty()
        try:
            results = self.model(
                frame,
                classes=[0],
                conf=0.5,
                imgsz=416,          # reduced from 640 — faster
                half=self.device == 'cuda',  # FP16 only on GPU
                device=self.device,
                verbose=False
            )
            detections = sv.Detections.from_ultralytics(results[0])
            if len(detections) > 0:
                detections = self.tracker.update_with_detections(detections)
            return detections
        except Exception as e:
            print(f'Detection error: {e}')
            return sv.Detections.empty()

detector = PersonDetector()