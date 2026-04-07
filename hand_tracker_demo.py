import time
import keyboard
from camera_wrapper import CameraWrapper
from hand_tracker import HandTracker
from hand_visualizer import HandVisualizer

camera = CameraWrapper(camera_index=1)
tracker = HandTracker("./hand_landmarker.task")
visualizer = HandVisualizer("/// hand tracker ///")

start_time = time.time()

while not keyboard.is_pressed('q'):
    img = camera()
    if img is None:
        continue
    ts = int((time.time() - start_time) * 1000)
    hand = tracker(img, ts)
    if hand.shape[0] == 0:
        continue
    visualizer.update(hand[0])

tracker.close()
camera.release()
visualizer.close()
