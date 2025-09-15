#!/usr/bin/env python3
"""
Check how YOLO mapped the class indices
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# Load the model
model_path = Path("models/unified_detector.pt")
model = YOLO(str(model_path))

# Check model's class names (if available)
print("Model info:")
print(f"Model path: {model_path}")

# Test with a sample from each folder
test_images = {
    "hand": "data/hand_cls/train/hand/hand_0001.jpg",
    "arm": "data/hand_cls/train/arm/arm_0001.jpg",
    "not_hand": "data/hand_cls/train/not_hand/not_hand_0001.jpg"
}

print("\nTesting known images:")
print("-" * 50)

for true_class, img_path in test_images.items():
    if Path(img_path).exists():
        # Run inference
        results = model(img_path, verbose=False)
        probs = results[0].probs

        # Get predictions
        top_class_idx = probs.top1
        confidence = float(probs.top1conf)

        # Show all probabilities
        print(f"\nTrue class: {true_class}")
        print(f"Image: {img_path}")
        print(f"Predicted class index: {top_class_idx}")
        print(f"Confidence: {confidence:.2%}")
        print("All probabilities:")
        print(f"  Index 0: {float(probs.data[0]):.2%}")
        print(f"  Index 1: {float(probs.data[1]):.2%}")
        print(f"  Index 2: {float(probs.data[2]):.2%}")

print("\n" + "=" * 50)
print("YOLO class mapping (based on alphabetical order):")
print("  Index 0 -> arm")
print("  Index 1 -> hand")
print("  Index 2 -> not_hand")
print("\nBUT in our code we're using:")
print("  classes = ['hand', 'arm', 'not_hand']")
print("\nThis is the problem! The indices are mismatched!")