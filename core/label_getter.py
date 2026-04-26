import keyboard

class LabelGetter:
    """
    Reads ground truth labels from live keyboard state.

    Args:
        label_keys (list[str]): monitored key names, e.g. ['w', 'a', 's', 'd'].
    """

    def __init__(self, label_keys):
        self.label_keys = label_keys

    def get_label(self):
        """
        Returns the index of the first currently-pressed monitored key.

        Returns:
            int or None: index into label_keys, or None if no key is held.
        """
        for i, key in enumerate(self.label_keys):
            if keyboard.is_pressed(key):
                return i
        return None
