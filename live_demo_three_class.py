#!/usr/bin/env python3
"""
Live webcam demo for three-class detection (hand/arm/not_hand)
Shows detection with color coding:
- RED: Hand detected
- YELLOW: Arm detected
- GRAY: Neither
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def main(weights='models/three_class_detector.pt', imgsz=224):
    # Check model exists
    if not Path(weights).exists():
        print(f"❌ Model not found: {weights}")
        print("   Train first with: python3 train_three_class.py")
        return

    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🖥️  Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("🎮 Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("💻 Using CPU")

    # Load model
    print(f"📦 Loading three-class model: {weights}")
    model = YOLO(weights)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam. Check your camera connection.")

    print("\n✅ Webcam started!")
    print("\nDetection colors:")
    print("  🔴 RED = Hand")
    print("  🟡 YELLOW = Arm")
    print("  ⚫ GRAY = Neither")
    print("\nPress 'q' to quit\n")

    # Frame counter for FPS
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for better UX
        frame = cv2.flip(frame, 1)

        # Resize for model
        resized = cv2.resize(frame, (imgsz, imgsz))

        # Run inference
        results = model(resized, verbose=False, device=device)[0]

        # Get predictions
        probs = results.probs.data.tolist()
        names = results.names
        top_idx = probs.index(max(probs))
        label = names[top_idx]
        confidence = max(probs) * 100

        # Visual feedback based on detection
        if label == 'hand':
            # HAND - RED
            text_color = (0, 0, 255)
            box_color = (0, 0, 255)
            detection_text = '✋ HAND DETECTED'
            thickness = 3

        elif label == 'arm':
            # ARM - YELLOW/ORANGE
            text_color = (0, 165, 255)
            box_color = (0, 165, 255)
            detection_text = '💪 ARM DETECTED'
            thickness = 3

        else:  # not_hand
            # NEITHER - GRAY
            text_color = (128, 128, 128)
            box_color = (128, 128, 128)
            detection_text = 'No hand or arm'
            thickness = 1

        # Draw border around frame for strong detections
        if confidence >= 75.0 and label != 'not_hand':
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                         box_color, thickness)

        # Display detection text
        cv2.putText(frame, detection_text, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

        # Show confidence
        cv2.putText(frame, f'Confidence: {confidence:.1f}%', (50, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Show all class probabilities
        y_offset = 130
        for i, (class_name, prob) in enumerate(zip(names.values(), probs)):
            prob_percent = prob * 100
            bar_color = text_color if i == top_idx else (200, 200, 200)

            # Class name
            cv2.putText(frame, f'{class_name}:', (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1, cv2.LINE_AA)

            # Probability bar
            bar_width = int(prob_percent * 2)
            cv2.rectangle(frame, (150, y_offset - 10), (150 + bar_width, y_offset),
                         bar_color, -1)
            cv2.rectangle(frame, (150, y_offset - 10), (350, y_offset),
                         bar_color, 1)

            # Percentage
            cv2.putText(frame, f'{prob_percent:.1f}%', (360, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1, cv2.LINE_AA)

            y_offset += 25

        # Show frame counter
        frame_count += 1
        cv2.putText(frame, f'Frame: {frame_count}', (frame.shape[1]-150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display
        cv2.imshow('Three-Class Detection - Live Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n👋 Goodbye! Processed {frame_count} frames")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live three-class detection demo')
    parser.add_argument('--weights', type=str, default='models/three_class_detector.pt',
                       help='Path to trained weights')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Image size for model (default: 224)')
    args = parser.parse_args()

    main(args.weights, args.imgsz)