# Hand Sign Detection Project 🖐️

A computer vision project for detecting hands and recognizing hand gestures using YOLO.

## Project Overview

This project implements a two-phase approach:
1. **Phase 1**: Hand detection - Detect hands in images
2. **Phase 2**: Gesture recognition - Classify specific hand gestures (👌, 👍, ✌️, etc.)

## Quick Start

### 1. Collect Training Data

```bash
# Start interactive data collection
python collect_data.py

# Check collected data statistics
python collect_data.py --stats
```

Collect 50-100 images per gesture with varied:
- Hand positions and angles
- Distances from camera
- Lighting conditions
- Backgrounds

### 2. Train Models

```bash
# Phase 1: Train hand detector
python train_hand_detector.py hand --epochs 30

# Phase 2: Train gesture classifier (after hand detector is ready)
python train_hand_detector.py gesture --epochs 50

# Or train both phases
python train_hand_detector.py both
```

### 3. Deploy to Hugging Face

The project includes a ready-to-deploy Gradio app for Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your trained models to `models/` directory
3. Copy `deployment/huggingface/` contents to your Space
4. The app will automatically load your models

## Project Structure

```
hand-sign-detection/
├── collect_data.py           # Webcam data collection tool
├── train_hand_detector.py    # Training pipeline
├── models/                   # Trained models (auto-created)
│   ├── hand_detector_v1.pt
│   └── gesture_classifier_v1.pt
├── data/                     # Training data
│   └── raw/                  # Collected images organized by gesture
│       ├── ok/
│       ├── thumbs_up/
│       ├── peace/
│       └── ...
├── deployment/
│   └── huggingface/         # Hugging Face deployment
│       ├── app.py           # Gradio interface
│       └── requirements.txt
└── claude.scratchpad.md     # Experiment tracking
```

## Supported Gestures

- 👌 OK sign (`ok`)
- 👍 Thumbs up (`thumbs_up`)
- ✌️ Peace sign (`peace`)
- ✊ Fist (`fist`)
- 👉 Pointing (`point`)
- 🤘 Rock sign (`rock`)
- 👋 Wave (`wave`)
- ✋ Stop (`stop`)
- 🖐️ Open hand (`hand`)
- Background/No hand (`none`)

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- `ultralytics` - YOLO implementation
- `opencv-python` - Image processing
- `gradio` - Web interface
- `torch` - Deep learning framework

## Web Interface

Once deployed to Hugging Face, your model will have:
- Live webcam input
- Image upload
- Real-time hand detection with bounding boxes
- Gesture classification with confidence scores
- Interactive demo interface

## Tips for Best Results

1. **Data Quality**: More diverse data > more epochs
2. **Balanced Dataset**: Collect similar amounts for each gesture
3. **Include Negatives**: Collect "none" class (no hands) to reduce false positives
4. **Test Incrementally**: Train for few epochs first to validate approach
5. **Monitor Training**: Watch for overfitting (val loss increasing)

## Next Steps

- [ ] Add more gesture types
- [ ] Implement real-time video processing
- [ ] Add hand tracking (not just detection)
- [ ] Create mobile app version
- [ ] Add gesture sequence recognition

## License

MIT