import keyboard

class LabelGetter:
    """
    Reads ground truth labels from keyboard key presses.
    Returns the index of the first pressed key, or None if no monitored key is held.
    """

    def __init__(self, label_keys):
        """
        Args:
            label_keys: list of key name strings, e.g. ['w', 'a', 's', 'd', 'shift']
        """
        self.label_keys = label_keys

    def get_label(self):
        """Returns the index of the first pressed key, or None."""
        for i, key in enumerate(self.label_keys):
            if keyboard.is_pressed(key):
                return i
        return None
