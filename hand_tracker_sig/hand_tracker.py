import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_TOPOLOGY = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def _to_bone_offsets(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts absolute landmark coordinates to parent-relative bone offsets.

    Wrist (joint 0) is set to (0, 0, 0). Each other joint becomes its vector
    offset from its parent, making the representation position-independent.

    Args:
        tensor (torch.Tensor): (n, 21, 3) absolute landmark coordinates.

    Returns:
        torch.Tensor: (n, 21, 3) bone offset representation.
    """
    out = tensor.clone()
    for h in range(out.shape[0]):
        out[h, 0] = 0.0
        for parent, child in _TOPOLOGY:
            out[h, child] = tensor[h, child] - tensor[h, parent]
    return out


class HandTracker:
    """
    MediaPipe-based hand landmark detector running in VIDEO mode.

    Returns bone-offset tensors so downstream models receive a
    position-independent representation of hand shape.

    Args:
        model_path (str): path to the hand_landmarker.task file.
        max_num_hands (int): maximum number of hands to detect per frame.
        min_hand_detection_confidence (float): min score to detect a new hand.
        min_hand_presence_confidence (float): min score to keep tracking a hand.
        min_tracking_confidence (float): min score for landmark tracking.
    """

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
        """Closes and reinitialises the landmarker, resetting internal tracking state."""
        self.landmarker.close()
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def __call__(self, bgr_frame: np.ndarray, timestamp_ms: int) -> torch.Tensor:
        """
        Detects hands in a frame and returns bone-offset tensors.

        Args:
            bgr_frame (np.ndarray): BGR image from the camera.
            timestamp_ms (int): frame timestamp in milliseconds.

        Returns:
            torch.Tensor: (n, 21, 3) bone offsets for n detected hands,
                or empty (0, 21, 3) tensor if no hands are found.
        """
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
        return _to_bone_offsets(torch.from_numpy(out))

    def close(self):
        """Releases the underlying MediaPipe landmarker."""
        self.landmarker.close()
