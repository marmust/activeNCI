import cv2
import numpy as np
import torch

_TOPOLOGY = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm knuckle row
    (2, 5),                                  # Thumb-index web
    (1, 5),                                  # Thumb diagonal
]

_BONE_COLOR  = (255,  60,   0)   # bright blue (BGR)
_JOINT_COLOR = (180,  30,   0)   # dimmer blue (BGR)
_JOINT_RADIUS = 4
_BONE_THICKNESS = 2


class HandVisualizer:
    """
    Live hand skeleton renderer backed by an OpenCV window.

    Each instance owns one named window, so multiple instances display
    side-by-side without interfering.

    Usage
    -----
        viz = HandVisualizer('Left hand')
        # inside your loop:
        viz.update(hand_tensor)   # (N, 21, 3) or (21, 3), x/y in [0, 1]
        # when done:
        viz.close()

    Or as a context manager:
        with HandVisualizer('Right hand') as viz:
            viz.update(tensor)
    """

    def __init__(self, window_name: str = 'Hand', size: tuple[int, int] = (400, 400)):
        """
        Args:
            window_name: Title of the OpenCV window. Must be unique per instance.
            size:        (width, height) of the canvas in pixels.
        """
        self.window_name = window_name
        self.w, self.h   = size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.w, self.h)

    # ------------------------------------------------------------------

    def update(self, hand_tensor) -> None:
        """
        Render one frame and push it to the window.

        Args:
            hand_tensor: Tensor / array of shape (N, 21, 3) or (21, 3).
                         x and y should be normalized to [0, 1] (raw MediaPipe
                         landmark coords). z is ignored.
        """
        pts = hand_tensor.cpu().numpy() if isinstance(hand_tensor, torch.Tensor) \
              else np.asarray(hand_tensor, dtype=np.float32)

        if pts.ndim == 2:          # (21, 3) → (1, 21, 3)
            pts = pts[np.newaxis]

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for hand in pts:
            px = np.clip((hand[:, 0] * (self.w - 1)).astype(int), 0, self.w - 1)
            py = np.clip((hand[:, 1] * (self.h - 1)).astype(int), 0, self.h - 1)

            for parent, child in _TOPOLOGY:
                cv2.line(canvas,
                         (px[parent], py[parent]),
                         (px[child],  py[child]),
                         _BONE_COLOR, _BONE_THICKNESS, cv2.LINE_AA)

            for x, y in zip(px, py):
                cv2.circle(canvas, (x, y), _JOINT_RADIUS,
                           _JOINT_COLOR, -1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Destroy this instance's window."""
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
