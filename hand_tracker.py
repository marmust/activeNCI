import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, model_path, max_num_hands=2,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        base = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def reset(self):
        self.landmarker.close()
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def __call__(self, bgr_frame: np.ndarray, timestamp_ms: int) -> torch.Tensor:
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            timestamp_ms=timestamp_ms,
        )
        if not result.hand_landmarks:
            return torch.empty((0, 21, 3), dtype=torch.float32)
        out = np.zeros((len(result.hand_landmarks), 21, 3), dtype=np.float32)
        for h, hand in enumerate(result.hand_landmarks):
            for i, lm in enumerate(hand):
                out[h, i] = [lm.x, lm.y, lm.z]
        return torch.from_numpy(out)

    def close(self):
        self.landmarker.close()
