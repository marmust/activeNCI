import ctypes
import ctypes.wintypes
import cv2
import numpy as np
import torch

def _make_drag_callback(window_name):
    """Returns a mouse callback that lets the user drag the named window."""
    state = {'dragging': False, 'abs0': (0, 0), 'win0': (0, 0)}

    def _abs_mouse():
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _win_pos():
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        r = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
        return r.left, r.top

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(dragging=True, abs0=_abs_mouse(), win0=_win_pos())
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if state['dragging']:
                ax, ay = _abs_mouse()
                cv2.moveWindow(window_name,
                               state['win0'][0] + ax - state['abs0'][0],
                               state['win0'][1] + ay - state['abs0'][1])
        elif event == cv2.EVENT_LBUTTONUP:
            state['dragging'] = False

    return _cb


# kinematic chain: parent-child bone pairs used to reconstruct absolute positions from offsets
_KINEMATIC = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]

# draw topology: kinematic chain plus palm knuckle row and thumb web lines
_DRAW_TOPOLOGY = [
    *_KINEMATIC,
    (5, 9), (9, 13), (13, 17),  # palm knuckle row
    (2, 5),                      # thumb-index web
    (1, 5),                      # thumb diagonal
]

_BONE_COLOR     = (255, 0, 0)   # BGR blue
_JOINT_COLOR    = (255, 0, 0)   # BGR blue
_JOINT_RADIUS   = 4
_BONE_THICKNESS = 2


class HandVisualizer:
    """
    Live hand skeleton renderer backed by an OpenCV window.

    Accepts bone-offset tensors, reconstructs absolute joint positions via the
    kinematic chain, centers on the mean joint, and stabilises scale with an EMA.

    Args:
        window_name (str): title of the OpenCV window.
        size (tuple[int, int]): (width, height) of the canvas in pixels.
        scale_ema (float): EMA smoothing factor for scale (higher = slower adaptation).
    """

    def __init__(self, window_name: str = 'Hand', size: tuple[int, int] = (400, 400),
                 scale_ema: float = 0.95):
        self.window_name = window_name
        self.w, self.h   = size
        self._scale_ema  = scale_ema
        self._running_range = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.w, self.h)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1.0)
        cv2.setMouseCallback(self.window_name, _make_drag_callback(self.window_name))

    def update(self, hand_tensor) -> None:
        """
        Renders one frame of bone-offset data and pushes it to the window.

        Args:
            hand_tensor: (n, 21, 3) or (21, 3) bone offsets. accepts both
                torch.Tensor and numpy arrays.
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

            # center on mean joint position
            absolute -= absolute.mean(axis=0)

            # stabilise scale via EMA to prevent jitter between frames
            max_range = np.abs(absolute[:, :2]).max()
            if max_range < 1e-6:
                max_range = 1.0
            if self._running_range is None:
                self._running_range = max_range
            else:
                self._running_range = (self._scale_ema * self._running_range
                                       + (1 - self._scale_ema) * max_range)
            scale = (min(self.w, self.h) * 0.4) / self._running_range

            # project to pixel coords centered on canvas
            px = np.clip((absolute[:, 0] * scale + self.w / 2).astype(int), 0, self.w - 1)
            py = np.clip((absolute[:, 1] * scale + self.h / 2).astype(int), 0, self.h - 1)

            # draw bones
            for parent, child in _DRAW_TOPOLOGY:
                cv2.line(canvas,
                         (px[parent], py[parent]),
                         (px[child],  py[child]),
                         _BONE_COLOR, _BONE_THICKNESS, cv2.LINE_AA)

            # draw joints as filled squares at 80% opacity, then add a 1px outline
            overlay = canvas.copy()
            for x, y in zip(px, py):
                r = int(_JOINT_RADIUS * 1.3)
                cv2.rectangle(overlay, (x - r, y - r), (x + r, y + r), _JOINT_COLOR, -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

            for x, y in zip(px, py):
                r = int(_JOINT_RADIUS * 1.3)
                cv2.rectangle(canvas, (x - r, y - r), (x + r, y + r), _JOINT_COLOR, 1)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        """Closes the OpenCV window."""
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
