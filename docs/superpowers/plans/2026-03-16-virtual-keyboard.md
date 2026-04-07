# Virtual Keyboard Demo — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a virtual keyboard demo that tracks two hands and maps individual finger gestures to keyboard keys, trained by the user simply typing on a real keyboard.

**Architecture:** Two independent per-hand pipelines (each reusing the existing AE + OKM + labeler stack unchanged) fed by a handedness-aware hand tracker. A new notebook orchestrates the dual pipelines, routes labels by QWERTY hand assignment, and renders a visual keyboard overlay.

**Tech Stack:** Same as existing — PyTorch, MediaPipe, OpenCV, `keyboard` library. No new dependencies.

---

## Current Architecture Assessment

The existing component stack is almost entirely reusable as-is:

| Component | Reusable? | Change Needed |
|---|---|---|
| `hand_tracker.py` | Mostly | Must expose handedness (Left/Right) from MediaPipe result |
| `autoencoder.py` | 100% | None — signal_shape=(1,21,3) works per-hand |
| `online_k_means.py` | 100% | None — just instantiate with more centroids |
| `live_dataset.py` | 100% | None — just instantiate with more classes |
| `centroid_labeler.py` | 100% | None — just instantiate with more labels |
| `ae_trainer.py` | 100% | None |
| `hand_visualizer.py` | 100% | None — already handles (N,21,3) |
| `camera_wrapper.py` | 100% | None |
| `label_getter.py` | Mostly | Needs to handle all alpha keys, not just 5 |

**Total changes to existing files: 1 (hand_tracker.py only) — additive, non-breaking. LabelGetter already accepts any key list.**

---

## Key Design Decisions

### 1. Per-Hand Independent Pipelines (not concatenated)

Each hand gets its own AE → OKM → Labeler pipeline operating on `(1,21,3)` tensors. This is better than concatenating to `(2,21,3)` because:
- Works when one hand is temporarily out of frame
- Reuses all existing components at their current signal shape — zero architectural changes
- Each hand only needs to learn ~13-15 keys, not 26+
- Halves the class space each pipeline must discriminate

### 2. QWERTY Hand Assignment for Label Routing

When the user presses a key during training, we know which hand it belongs to from standard touch-typing layout. This avoids having to detect which hand "moved" — we just route the label to the correct hand's pipeline.

**Left hand keys (15):** `q w e r t a s d f g z x c v b`
**Right hand keys (14):** `y u i o p h j k l n m , . space`

(Adjustable — the user might want a different split or fewer keys initially.)

### 3. Centroid Count Per Hand

With ~15 keys per hand, we need at least 15 centroids. Using **18 centroids per hand** gives headroom for dead centroids and sub-gesture variations. The k-means dead-centroid reinit logic already handles this well.

### 4. Training UX

Same as existing demo — user holds a key while performing the gesture, system records the label. The difference: instead of 5 keys, it's all alpha keys. The user just types normally (without moving wrists) and the system learns.

A **guided training mode** (optional enhancement): display a target character on screen, user types it repeatedly. This is the new notebook's addition, not a library change.

### 5. Autoencoder Bottleneck

Keep at **32** — the hand shape with stationary wrist is primarily encoded by finger curls and spreads. 32 dims is plenty to distinguish 15 finger positions. If discrimination proves weak, bump to 48.

---

## File Plan

### Files to Modify

1. **`hand_tracker.py`** — Add handedness extraction from MediaPipe result
   - Add `_last_handedness` attribute
   - Populate it in `__call__` from `result.handedness`
   - Add `handedness` property
   - Existing callers unaffected (return value doesn't change)

2. **`label_getter.py`** — Generalize to accept arbitrary key lists
   - Currently works fine already — just needs to be instantiated with a longer `label_keys` list
   - Actually no code change needed! Just pass all 26+ keys at construction time
   - **Correction: no change needed.** LabelGetter already accepts any `label_keys` list.

### Files to Create

3. **`keyboard_layout.py`** (~40 lines) — QWERTY key mapping and hand assignment
   - `LEFT_KEYS`, `RIGHT_KEYS` lists
   - `ALL_KEYS` combined list
   - `hand_for_key(key) -> 'Left' | 'Right'` helper
   - `key_index_for_hand(key, hand) -> int` — returns label index within that hand's key set
   - `KEYBOARD_ROWS` — for rendering the visual keyboard layout

4. **`keyboard_visualizer.py`** (~80 lines) — OpenCV keyboard overlay renderer
   - Renders a QWERTY keyboard layout on a canvas
   - Highlights the currently predicted key
   - Shows per-key confidence
   - Color-codes left/right hand keys

5. **`virtual_keyboard.ipynb`** — Main demo notebook (~6 cells, same structure as `rewrite.ipynb`)

---

## Chunk 1: Hand Tracker Handedness + Keyboard Layout

### Task 1: Add handedness to HandTracker

**Files:**
- Modify: `hand_tracker.py:27-59`

- [ ] **Step 1: Add handedness tracking to HandTracker**

In `__init__`, add:
```python
self._last_handedness = []  # populated each frame: ['Left', 'Right'] etc.
```

In `__call__`, after `result = self.landmarker.detect_for_video(...)`, before the `if not result.hand_landmarks:` check, add nothing yet. After the existing `if not result.hand_landmarks:` early return, add handedness extraction:

```python
# right before the return at the end of __call__:
self._last_handedness = [
    h[0].category_name for h in result.handedness
]
```

Also reset in the empty case:
```python
if not result.hand_landmarks:
    self._last_handedness = []
    return torch.empty((0, 21, 3), dtype=torch.float32)
```

Also reset in `reset()`:
```python
def reset(self):
    self.landmarker.close()
    self._last_handedness = []
    self.landmarker = vision.HandLandmarker.create_from_options(self.options)
```

Add property:
```python
@property
def handedness(self):
    """List of handedness strings from last detection, e.g. ['Left', 'Right']."""
    return self._last_handedness
```

- [ ] **Step 2: Verify existing demo still works**

Run `rewrite.ipynb` — it only uses `hand_tracker(img, ts)` return value, never `.handedness`. Should be fully backward compatible.

- [ ] **Step 3: Commit**

```bash
git add hand_tracker.py
git commit -m "feat: expose handedness from MediaPipe detection in HandTracker"
```

---

### Task 2: Create keyboard_layout.py

**Files:**
- Create: `keyboard_layout.py`

- [ ] **Step 1: Write keyboard_layout.py**

```python
"""QWERTY keyboard layout with left/right hand assignment for virtual keyboard demo."""

# Standard touch-typing hand assignment
LEFT_KEYS  = list('qwertasdfgzxcvb')   # 15 keys
RIGHT_KEYS = list('yuiophjklnm')        # 11 keys  (can add ,./; later)

ALL_KEYS = LEFT_KEYS + RIGHT_KEYS

# For rendering — each row is a list of (key, hand) tuples
KEYBOARD_ROWS = [
    [(k, 'Left' if k in LEFT_KEYS else 'Right') for k in 'qwertyuiop'],
    [(k, 'Left' if k in LEFT_KEYS else 'Right') for k in 'asdfghjkl'],
    [(k, 'Left' if k in LEFT_KEYS else 'Right') for k in 'zxcvbnm'],
]


def hand_for_key(key: str) -> str:
    """Returns 'Left' or 'Right' for a given key."""
    if key in LEFT_KEYS:
        return 'Left'
    if key in RIGHT_KEYS:
        return 'Right'
    raise ValueError(f"Unknown key: {key!r}")


def key_index(key: str, hand: str) -> int:
    """Returns the label index for a key within its hand's key set."""
    keys = LEFT_KEYS if hand == 'Left' else RIGHT_KEYS
    return keys.index(key)
```

- [ ] **Step 2: Commit**

```bash
git add keyboard_layout.py
git commit -m "feat: add QWERTY keyboard layout with hand assignments"
```

---

## Chunk 2: Keyboard Visualizer

### Task 3: Create keyboard_visualizer.py

**Files:**
- Create: `keyboard_visualizer.py`

- [ ] **Step 1: Write keyboard_visualizer.py**

This renders a QWERTY keyboard layout in an OpenCV window with:
- Each key as a rectangle
- Color-coded by hand (blue=left, green=right)
- The predicted key highlighted brightly
- Confidence shown as brightness/opacity

```python
import cv2
import numpy as np
from keyboard_layout import KEYBOARD_ROWS, LEFT_KEYS

# Layout constants
KEY_W, KEY_H = 50, 50
KEY_PAD = 4
ROW_OFFSETS = [0, 25, 50]  # QWERTY stagger (pixels)

LEFT_COLOR  = (180, 80, 40)    # blue-ish (BGR)
RIGHT_COLOR = (40, 140, 80)    # green-ish (BGR)
HIGHLIGHT   = (0, 255, 255)    # yellow highlight for predicted key
BG_COLOR    = (30, 30, 30)


class KeyboardVisualizer:
    """Renders a QWERTY keyboard overlay showing predicted keys."""

    def __init__(self, window_name='Virtual Keyboard'):
        self.window_name = window_name
        # compute canvas size from layout
        max_row_len = max(len(row) for row in KEYBOARD_ROWS)
        max_offset = max(ROW_OFFSETS[:len(KEYBOARD_ROWS)])
        self.w = max_row_len * (KEY_W + KEY_PAD) + max_offset + KEY_PAD
        self.h = len(KEYBOARD_ROWS) * (KEY_H + KEY_PAD) + KEY_PAD
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.w, self.h)

    def update(self, predicted_key=None):
        """
        Render keyboard with highlighting on the predicted key.

        Args:
            predicted_key: str or None — the key currently predicted
        """
        canvas = np.full((self.h, self.w, 3), BG_COLOR, dtype=np.uint8)

        for row_idx, row in enumerate(KEYBOARD_ROWS):
            for col_idx, (key, hand) in enumerate(row):
                x = ROW_OFFSETS[row_idx] + col_idx * (KEY_W + KEY_PAD) + KEY_PAD
                y = row_idx * (KEY_H + KEY_PAD) + KEY_PAD

                # base color by hand
                color = LEFT_COLOR if key in LEFT_KEYS else RIGHT_COLOR

                # highlight predicted key
                if key == predicted_key:
                    color = HIGHLIGHT

                cv2.rectangle(canvas, (x, y), (x + KEY_W, y + KEY_H), color, -1)
                cv2.rectangle(canvas, (x, y), (x + KEY_W, y + KEY_H), (80, 80, 80), 1)

                # draw key label
                cv2.putText(canvas, key.upper(), (x + 15, y + 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
```

- [ ] **Step 2: Commit**

```bash
git add keyboard_visualizer.py
git commit -m "feat: add QWERTY keyboard visualizer with predicted key highlighting"
```

---

## Chunk 3: Virtual Keyboard Demo Notebook

### Task 4: Create virtual_keyboard.ipynb

**Files:**
- Create: `virtual_keyboard.ipynb`

This is the main deliverable. It follows the exact same structure as `rewrite.ipynb` but with dual pipelines.

- [ ] **Step 1: Write Cell 0 — Imports**

```python
import torch
import time
import keyboard
from IPython.display import clear_output

from camera_wrapper import CameraWrapper
from hand_tracker import HandTracker
from online_k_means import OnlineKmeans
from live_dataset import LiveDataset
from autoencoder import HumanSignalAutoencoder
from ae_trainer import AETrainer
from hand_visualizer import HandVisualizer
from centroid_labeler import CentroidLabeler
from label_getter import LabelGetter
from keyboard_layout import LEFT_KEYS, RIGHT_KEYS, ALL_KEYS, hand_for_key, key_index
from keyboard_visualizer import KeyboardVisualizer
```

- [ ] **Step 2: Write Cell 1 — Settings**

```python
# architecture settings (per hand — each hand is an independent pipeline)
human_signal_shape = (1, 21, 3)  # single hand: 21 joints * 3 coords
autoencoder_bottleneck = 32

# clustering: enough centroids to cover ~15 keys per hand + headroom
okm_num_centroids_left  = 18
okm_num_centroids_right = 15  # fewer keys on right hand

# live dataset
queue_size_per_class = 200

# autoencoder training
lr = 0.0001
ae_batch_size = 32
ae_steps_per_frame = 10

# labels — each hand has its own label set
num_labels_left  = len(LEFT_KEYS)   # 15
num_labels_right = len(RIGHT_KEYS)  # 11

# label assignment confidence gate
label_margin_threshold = 0.75

# MediaPipe handedness mirroring: front-facing cameras often report
# Left/Right swapped. Set True to flip handedness labels.
swap_handedness = True

# OKM retrain frequency: retrain every N frames (saves ~50% compute vs every frame)
okm_retrain_every = 2

# control keys (these are NOT part of the trainable key set)
quit_key = 'esc'
autopress_toggle_key = 'tab'

# torch
run_device = torch.device("cuda")
```

- [ ] **Step 3: Write Cell 2 — Initialize shared components**

```python
camera = CameraWrapper(camera_index=1)
hand_tracker = HandTracker("./hand_landmarker.task", max_num_hands=2)
```

- [ ] **Step 4: Write Cell 3 — Initialize per-hand pipelines**

```python
# --- LEFT HAND PIPELINE ---
left_ae = HumanSignalAutoencoder(human_signal_shape, autoencoder_bottleneck).to(run_device)
left_okm = OnlineKmeans(okm_num_centroids_left, run_device)
left_dataset = LiveDataset(okm_num_centroids_left, queue_size_per_class)
left_ae_trainer = AETrainer(left_ae, lr=lr, device=run_device)
left_labeler = CentroidLabeler(okm_num_centroids_left, num_labels_left)

# --- RIGHT HAND PIPELINE ---
right_ae = HumanSignalAutoencoder(human_signal_shape, autoencoder_bottleneck).to(run_device)
right_okm = OnlineKmeans(okm_num_centroids_right, run_device)
right_dataset = LiveDataset(okm_num_centroids_right, queue_size_per_class)
right_ae_trainer = AETrainer(right_ae, lr=lr, device=run_device)
right_labeler = CentroidLabeler(okm_num_centroids_right, num_labels_right)

# label getters — one per hand, watching only that hand's keys
left_label_getter = LabelGetter(LEFT_KEYS)
right_label_getter = LabelGetter(RIGHT_KEYS)

# visualizers
left_hand_viz = HandVisualizer('Left Hand')
right_hand_viz = HandVisualizer('Right Hand')
keyboard_viz = KeyboardVisualizer('Virtual Keyboard')
```

- [ ] **Step 5: Write Cell 4 — Helper to process one hand through its pipeline**

This avoids duplicating the pipeline logic:

```python
def process_hand(signal, ae, okm, dataset, ae_trainer_obj, labeler, label_getter_obj,
                 label_map, confidence_map, device, should_retrain_okm):
    """
    Process a single hand through its pipeline.
    Returns (class_idx, label_map, confidence_map, loss_val, predicted_label_idx).
    """
    signal_dev = signal.to(device)

    # encode
    with torch.no_grad():
        latent = ae.encode(signal_dev.unsqueeze(0)).squeeze(0)

    # classify
    class_idx, dists = okm.classify(latent)
    if class_idx is None:
        class_idx = 0

    # get ground truth label
    label = label_getter_obj.get_label()
    label_key_pressed = label is not None

    # confidence gate
    if label is not None and dists is not None:
        sorted_dists = dists.sort().values
        margin = sorted_dists[0] / (sorted_dists[1] + 1e-8)
        if margin >= label_margin_threshold:
            label = None

    # store
    dataset.apply_datapoint(latent, signal_dev, class_idx, label=label)

    # train AE
    loss_val = float('nan')
    for _ in range(ae_steps_per_frame):
        batch = dataset.get_random_human_sigs(ae_batch_size)
        if batch is not None:
            loss_val = ae_trainer_obj.train_step(batch)

    # retrain OKM (skipped on some frames for performance)
    if should_retrain_okm:
        all_latents = dataset.get_random_latents(dataset.total_size())
        okm.retrain(all_latents)

    # update labels only when user is actively labeling
    if label_key_pressed:
        label_map = labeler.get_label_map(dataset)
        confidence_map = labeler.get_confidence_map(dataset)

    # predicted label
    predicted_label_idx = label_map.get(class_idx)

    return class_idx, label_map, confidence_map, loss_val, predicted_label_idx
```

- [ ] **Step 6: Write Cell 5 — Main loop**

```python
start_time = time.time()
autopress_enabled = False
toggle_was_pressed = False
active_key = None
frame_count = 0

left_label_map = {i: None for i in range(okm_num_centroids_left)}
left_confidence_map = {i: 0.0 for i in range(okm_num_centroids_left)}
right_label_map = {i: None for i in range(okm_num_centroids_right)}
right_confidence_map = {i: 0.0 for i in range(okm_num_centroids_right)}
left_loss = right_loss = float('nan')

while not keyboard.is_pressed(quit_key):

    img = camera()
    if img is None:
        continue

    frame_count += 1
    should_retrain_okm = (frame_count % okm_retrain_every == 0)

    ts = int((time.time() - start_time) * 1000)
    all_hands = hand_tracker(img, ts).to(run_device)
    handedness = hand_tracker.handedness

    # separate hands by handedness (swap if MediaPipe reports mirrored labels)
    left_signal = right_signal = None
    for i, h in enumerate(handedness):
        if swap_handedness:
            h = 'Right' if h == 'Left' else 'Left'
        if h == 'Left' and left_signal is None:
            left_signal = all_hands[i:i+1]  # keep (1, 21, 3) shape
        elif h == 'Right' and right_signal is None:
            right_signal = all_hands[i:i+1]

    predicted_key = None

    # --- process left hand ---
    if left_signal is not None:
        (l_cls, left_label_map, left_confidence_map,
         left_loss, l_pred) = process_hand(
            left_signal, left_ae, left_okm, left_dataset,
            left_ae_trainer, left_labeler, left_label_getter,
            left_label_map, left_confidence_map, run_device, should_retrain_okm
        )
        left_hand_viz.update(left_signal.detach().cpu())
        if l_pred is not None:
            predicted_key = LEFT_KEYS[l_pred]

    # --- process right hand ---
    if right_signal is not None:
        (r_cls, right_label_map, right_confidence_map,
         right_loss, r_pred) = process_hand(
            right_signal, right_ae, right_okm, right_dataset,
            right_ae_trainer, right_labeler, right_label_getter,
            right_label_map, right_confidence_map, run_device, should_retrain_okm
        )
        right_hand_viz.update(right_signal.detach().cpu())
        if r_pred is not None:
            predicted_key = RIGHT_KEYS[r_pred]

    # --- update keyboard visualizer ---
    keyboard_viz.update(predicted_key=predicted_key)

    # --- status display ---
    clear_output(wait=True)
    lh = 'detected' if left_signal is not None else 'missing'
    rh = 'detected' if right_signal is not None else 'missing'
    print(f"Hands: L={lh} R={rh} | Predicted: {predicted_key or '?'}")
    print(f"L loss: {left_loss:.6f} | R loss: {right_loss:.6f}")
    print(f"L queues: {left_dataset.get_queue_sizes()}")
    print(f"R queues: {right_dataset.get_queue_sizes()}")
    print(f"Auto: {'ON  key=' + (active_key or 'none') if autopress_enabled else 'OFF'}")

    # --- autopress ---
    toggle_is_pressed = keyboard.is_pressed(autopress_toggle_key)
    if toggle_is_pressed and not toggle_was_pressed:
        autopress_enabled = not autopress_enabled
        if not autopress_enabled and active_key is not None:
            keyboard.release(active_key)
            active_key = None
    toggle_was_pressed = toggle_is_pressed

    if autopress_enabled and predicted_key is not None:
        if predicted_key != active_key:
            if active_key is not None:
                keyboard.release(active_key)
            active_key = predicted_key
            keyboard.press(active_key)

if active_key is not None:
    keyboard.release(active_key)
```

- [ ] **Step 7: Commit**

```bash
git add virtual_keyboard.ipynb
git commit -m "feat: add virtual keyboard demo with dual-hand pipelines"
```

---

## Chunk 4: Tuning & Polish (Post-Integration)

### Task 5: Test and tune parameters

These are manual tuning steps to do after the first run:

- [ ] **Step 1: Verify hand separation works**

Run the notebook. Confirm that `hand_tracker.handedness` correctly returns `['Left', 'Right']` for two hands. MediaPipe sometimes mirrors (camera left = anatomical right). If labels are swapped, the fix is to flip the handedness check:

```python
# If MediaPipe reports mirrored handedness (common with front-facing camera):
# swap 'Left' ↔ 'Right' in the separation logic
```

- [ ] **Step 2: Tune centroid counts**

If many centroids stay dead (empty queues), reduce `okm_num_centroids_*`. If keys share centroids (low confidence), increase. Start with 18/15 and adjust.

- [ ] **Step 3: Tune label_margin_threshold**

The margin is `nearest_dist / second_nearest_dist` — close to 0 = confident, close to 1 = ambiguous. Labels are rejected when `margin >= threshold`. With more centroids, margins will be tighter (closer to 1.0), so you may need to **raise** the threshold (e.g., from `0.75` to `0.85` or `0.9`) to allow more labels through. Setting to `1.0` disables the gate entirely.

- [ ] **Step 4: Consider bottleneck size**

If the AE reconstruction loss plateaus high, try `autoencoder_bottleneck = 48`. This gives more capacity to encode subtle finger differences. The body_width calculation in the AE auto-adjusts: `(63 + 48) // 2 = 55`.

- [ ] **Step 5: Commit tuned parameters**

```bash
git add virtual_keyboard.ipynb
git commit -m "tune: adjust keyboard demo parameters after testing"
```

---

## Delta Summary

| Change | Type | Lines (est.) | Risk |
|---|---|---|---|
| `hand_tracker.py` — add handedness | Modify | +10 | Low — additive, non-breaking |
| `keyboard_layout.py` — key mapping | New file | ~35 | None — pure data |
| `keyboard_visualizer.py` — keyboard renderer | New file | ~80 | None — standalone display |
| `virtual_keyboard.ipynb` — demo notebook | New file | ~150 | Low — follows existing pattern |
| Existing components (AE, OKM, etc.) | **No change** | 0 | Zero |

**Total: ~275 lines of new/changed code. 1 existing file modified. 3 new files.**

---

## Risks & Mitigations

### 1. MediaPipe handedness mirroring
**Risk:** Front-facing cameras may report Left/Right swapped (anatomical vs camera perspective).
**Mitigation:** Add a `swap_handedness` boolean config in the notebook. Test on first run and flip if needed.

### 2. Too many keys → poor centroid separation
**Risk:** 15 keys per hand may be hard to discriminate with k-means in 32-dim latent space.
**Mitigation:** Start with fewer keys (home row only: `asdfg` / `hjkl`) and expand once that works. The code already supports arbitrary key lists.

### 3. Training time
**Risk:** 26 keys × ~200 samples each = much more training data needed than the 5-gesture demo.
**Mitigation:** Guided training mode — cycle through keys one at a time, collecting focused data per key. This can be a simple enhancement: display "Type 'a' 20 times" prompt.

### 4. Auto-press conflicts during training
**Risk:** User pressing real keys for labels while autopress tries to press predicted keys.
**Mitigation:** Already handled — autopress is toggled with `tab`, off by default. Train with it off, then toggle on.

### 5. Single hand disappears temporarily
**Risk:** Pipeline continues from stale state when hand re-enters frame.
**Mitigation:** Already handled — each pipeline only runs when its hand is detected. OKM and AE retain their trained state across frames where the hand is missing. No special handling needed.

---

## Phased Approach (Recommended)

If 26 keys feels too ambitious for the first pass:

**Phase 1 — Home row only (8 keys):**
- `LEFT_KEYS = list('asdf')`, `RIGHT_KEYS = list('jkl;')`
- `okm_num_centroids = 6` per hand
- Proves the dual-pipeline architecture works

**Phase 2 — Full alpha:**
- Expand to all 26 keys
- Tune centroids, bottleneck, margin threshold

**Phase 3 — Symbols & modifiers:**
- Add space, shift, numbers
- May need additional hand pose (e.g., thumb position for space bar)
