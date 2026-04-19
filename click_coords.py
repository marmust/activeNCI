from pynput import mouse

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Click: ({x}, {y})")

print("Click anywhere to print coords. Ctrl+C to quit.")
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
