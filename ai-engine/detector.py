import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np

class PersonDetector:
    def __init__(self):
        print('Loading YOLO11s model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        self.model = YOLO("yolo11s.pt")      # ← self.model
        self.model.to(self.device)            # ← self.model

        # Warmup run
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
                imgsz=640,
                device=self.device,    # ← force GPU here too
                verbose=False
            )
            return sv.Detections.from_ultralytics(results[0])
        except Exception as e:
            print(f'Detection error: {e}')
            return sv.Detections.empty()

detector = PersonDetector()
