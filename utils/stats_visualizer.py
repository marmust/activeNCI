import cv2
import numpy as np

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.52
_THICKNESS  = 1

class StatsVisualizer:
    """
    Auto-sizing OpenCV window that renders a list of text lines.

    Args:
        window_name (str): title of the OpenCV window.
        line_height (int): vertical spacing per line in pixels.
        padding (int): horizontal and vertical padding in pixels.
    """

    def __init__(self, window_name='StatsVisualizer', line_height=32, padding=12):
        self.window_name = window_name
        self.line_height = line_height
        self.padding = padding
        self._w = 0
        self._h = 0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1.0)

    def update(self, lines: list[str]) -> None:
        """
        Redraws the window with a new list of text lines.

        Resizes the window automatically to fit the longest line.

        Args:
            lines (list[str]): lines to display, top to bottom.
        """
        max_text_w = max(
            cv2.getTextSize(line, _FONT, _FONT_SCALE, _THICKNESS)[0][0]
            for line in lines
        ) if lines else 0
        w = max_text_w + self.padding * 2
        h = self.line_height * max(len(lines), 1) + self.padding * 2

        # resize only when dimensions change to avoid flicker
        if w != self._w or h != self._h:
            self._w, self._h = w, h
            cv2.resizeWindow(self.window_name, w, h)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            y = self.padding + self.line_height * i + self.line_height - 8
            cv2.putText(canvas, line, (self.padding, y),
                        _FONT, _FONT_SCALE, (200, 200, 200), _THICKNESS, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        """Closes the OpenCV window."""
        cv2.destroyWindow(self.window_name)
