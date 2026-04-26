# activeNCI

![banner](https://media.discordapp.net/attachments/990898405310074892/1497989353195045034/gsdfgssgtdged.png?ex=69ef86cd&is=69ee354d&hm=3d012b6301d52ace05aa5d66b20466b2fb51c91beca053900e1237b44d814164&=&format=webp&quality=lossless&width=2112&height=792)

active Neural Computer Interface or activeNCI is an AI system for interpreting a real time signal from a human (currently a hand tracker, meant to be an EEG) and classifying its current state. the innovation here is that the classification doesnt NEED labels, instead relying on online K means.

## How it works

Each frame, a MediaPipe hand tracker produces a bone-offset representation of the hand. A small autoencoder compresses this into a latent vector, which an online k-means classifier assigns to a gesture cluster. Holding a label key associates the current cluster with that key. The whole pipeline trains continuously in the background.

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)

```sh
pip install -r requirements.txt
```

## Setup

Download the MediaPipe hand landmark model from the [MediaPipe model card](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the project root as `hand_landmarker.task`.

## Usage

Open `activeNCI.ipynb` and run the cells top to bottom. Adjust the settings cell for your hardware (camera index, device, label keys). In the main loop, hold a label key while performing a gesture to train the classifier. Press `q` to stop.

## Structure

```yaml
core/                signal-agnostic ML pipeline
  autoencoder.py     encoder/decoder for arbitrary signal shapes
  online_k_means.py  online k-means++ with dead centroid reinit
  live_dataset.py    per-class rolling queue storage
  ae_trainer.py      AdamW training loop
  centroid_labeler.py centroid-to-label mapping with majority vote
  label_getter.py    keyboard-based ground truth capture

hand_tracker_sig/    MediaPipe hand tracking
  hand_tracker.py    landmark detection + bone offset conversion
  camera_wrapper.py  OpenCV camera input
  hand_tracker_demo.py minimal visualisation demo

utils/               visualizers
  hand_visualizer.py  bone skeleton renderer with EMA scale
  stats_visualizer.py auto-sizing text overlay window

activeNCI.ipynb      main demo notebook
```
