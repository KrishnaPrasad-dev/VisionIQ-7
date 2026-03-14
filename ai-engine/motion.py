import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self):
        # MOG2 for background subtraction (abandoned objects)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

        # Optical flow state
        self.prev_gray   = None
        self.frame_counter = 0

        # Abandoned object tracking
        self.stationary_regions = {}

        # Thresholds - tuned to reduce false positives
        self.RUNNING_SPEED_THRESHOLD    = 25.0  # pixels/frame
        self.ABANDONED_FRAMES_THRESHOLD = 150   # ~5 seconds at 30fps

    def analyze(self, frame):
        """
        Returns:
          running       bool  - fast movement detected
          panic         bool  - chaotic multi-direction movement
          abandoned     bool  - stationary object detected
          motion_score  float - 0-100 motion threat level
          flow_vectors  list  - for visualization
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        self.frame_counter += 1
        skip_optical_flow = (self.frame_counter % 3 != 0)  # run every 3rd frame

        running      = False
        panic        = False
        abandoned    = False
        motion_score = 0.0
        flow_vectors = []

        # ── OPTICAL FLOW (every 3rd frame) ───────────────────
        if self.prev_gray is not None and not skip_optical_flow:
            points = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=80,
                qualityLevel=0.3,
                minDistance=10,
                blockSize=7
            )

            if points is not None and len(points) > 5:
                curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    points,
                    None,
                    winSize=(15, 15),
                    maxLevel=2
                )

                good_prev = points[status == 1]
                good_curr = curr_points[status == 1]

                if len(good_prev) > 5:
                    vectors   = good_curr - good_prev
                    speeds    = np.linalg.norm(vectors, axis=1)
                    avg_speed = float(np.mean(speeds))

                    flow_vectors = list(zip(
                        good_prev.reshape(-1, 2).tolist(),
                        good_curr.reshape(-1, 2).tolist()
                    ))

                    # Running — high average speed
                    if avg_speed > self.RUNNING_SPEED_THRESHOLD:
                        running = True
                        motion_score += min(avg_speed * 1.5, 40)

                    # Panic — chaotic vectors in many directions
                    # Stricter: needs high speed AND high angle variance
                    if len(vectors) > 10:
                        angles    = np.arctan2(vectors[:, 1], vectors[:, 0])
                        angle_std = float(np.std(angles))
                        if angle_std > 1.5 and avg_speed > 12:
                            panic = True
                            motion_score += 30

        self.prev_gray = gray

        # ── MOG2 BACKGROUND SUBTRACTION ──────────────────────
        fg_mask = self.bg_subtractor.apply(frame)

        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_regions = set()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 50000:
                x, y, rw, rh = cv2.boundingRect(cnt)
                region_key   = (x // 20, y // 20)
                current_regions.add(region_key)

                if region_key not in self.stationary_regions:
                    self.stationary_regions[region_key] = 0
                self.stationary_regions[region_key] += 1

                if self.stationary_regions[region_key] > self.ABANDONED_FRAMES_THRESHOLD:
                    abandoned     = True
                    motion_score += 25

        # Clean up regions no longer present
        for key in list(self.stationary_regions.keys()):
            if key not in current_regions:
                del self.stationary_regions[key]

        motion_score = min(motion_score, 100)

        return {
            'running':      running,
            'panic':        panic,
            'abandoned':    abandoned,
            'motion_score': round(motion_score),
            'flow_vectors': flow_vectors
        }

# Single global instance
motion_analyzer = MotionAnalyzer()