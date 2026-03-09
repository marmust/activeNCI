import cv2
import numpy as np

class CameraWrapper:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f'Cannot open camera at index {camera_index}')

    def __call__(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.release()
