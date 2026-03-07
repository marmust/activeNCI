import cv2
import numpy as np
import torch

# kinematic chain — the 20 parent→child bones for reconstructing positions from offsets
_KINEMATIC = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# display topology — kinematic + extra palm/web lines for nicer rendering
_DRAW_TOPOLOGY = [
    *_KINEMATIC,
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

    Accepts bone-offset tensors (from to_bone_offsets): reconstructs absolute
    positions via the kinematic chain, centers on the mean of all joints,
    and uses a consistent scale via EMA so the hand size stays stable.
    """

    def __init__(self, window_name: str = 'Hand', size: tuple[int, int] = (400, 400),
                 scale_ema: float = 0.95):
        self.window_name = window_name
        self.w, self.h   = size
        self._scale_ema  = scale_ema     # smoothing factor for scale tracking
        self._running_range = None       # EMA of max coordinate range
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.w, self.h)

    def update(self, hand_tensor) -> None:
        """
        Render one frame of bone-offset data and push it to the window.

        Args:
            hand_tensor: (N, 21, 3) or (21, 3) bone offsets.
        """
        pts = hand_tensor.cpu().numpy() if isinstance(hand_tensor, torch.Tensor) \
              else np.asarray(hand_tensor, dtype=np.float32)

        if pts.ndim == 2:
            pts = pts[np.newaxis]

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for hand in pts:
            # reconstruct absolute positions by walking the kinematic chain
            absolute = hand.copy()
            for parent, child in _KINEMATIC:
                absolute[child] = absolute[parent] + hand[child]

            # center on the mean of all joints
            center = absolute.mean(axis=0)
            absolute -= center

            # consistent scale via EMA — prevents jittery resizing between frames
            max_range = np.abs(absolute[:, :2]).max()
            if max_range < 1e-6:
                max_range = 1.0
            if self._running_range is None:
                self._running_range = max_range
            else:
                self._running_range = self._scale_ema * self._running_range + (1 - self._scale_ema) * max_range
            scale = (min(self.w, self.h) * 0.4) / self._running_range

            # map to pixel coords centered on canvas
            px = np.clip((absolute[:, 0] * scale + self.w / 2).astype(int), 0, self.w - 1)
            py = np.clip((absolute[:, 1] * scale + self.h / 2).astype(int), 0, self.h - 1)

            for parent, child in _DRAW_TOPOLOGY:
                cv2.line(canvas,
                         (px[parent], py[parent]),
                         (px[child],  py[child]),
                         _BONE_COLOR, _BONE_THICKNESS, cv2.LINE_AA)

            for x, y in zip(px, py):
                cv2.circle(canvas, (x, y), _JOINT_RADIUS,
                           _JOINT_COLOR, -1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
