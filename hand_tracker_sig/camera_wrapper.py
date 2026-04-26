import cv2
import numpy as np

class CameraWrapper:
    """
    Thin wrapper around an OpenCV VideoCapture for frame-by-frame access.

    Args:
        camera_index (int): index of the camera to open.
    """

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f'cannot open camera at index {camera_index}')

    def __call__(self) -> np.ndarray | None:
        """
        Captures and returns the next frame.

        Returns:
            np.ndarray or None: BGR frame, or None if capture failed.
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Releases the underlying VideoCapture."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
