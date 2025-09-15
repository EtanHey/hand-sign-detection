#!/usr/bin/env python3
"""
Live webcam demo for hand detection with real-time visual feedback.
Shows confidence with color coding:
- RED: High confidence (85%+)
- GREEN: Lower confidence
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def main(weights='models/hand_detector.pt', imgsz=224):
    # Check model exists
    if not Path(weights).exists():
        print(f"❌ Model not found: {weights}")
        print("   Train first with: python3 train_hands_only.py")
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
    print(f"📦 Loading model: {weights}")
    model = YOLO(weights)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam. Check your camera connection.")

    print("\n✅ Webcam started!")
    print("Press 'q' to quit\n")

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
            # Determine color based on confidence
            if confidence >= 85.0:
                # HIGH CONFIDENCE - RED (exciting!)
                text_color = (0, 0, 255)
                box_color = (0, 0, 255)
                detection_text = '🖐️ HAND DETECTED!'
                thickness = 3
            else:
                # LOWER CONFIDENCE - GREEN
                text_color = (0, 255, 0)
                box_color = (0, 255, 0)
                detection_text = 'Hand detected'
                thickness = 2

            # Draw border around frame
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                         box_color, thickness)

            # Display detection text
            cv2.putText(frame, detection_text, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

            # Show confidence
            cv2.putText(frame, f'Confidence: {confidence:.1f}%', (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

            # Add visual indicator bar
            bar_width = int((confidence / 100) * 300)
            cv2.rectangle(frame, (50, 110), (50 + bar_width, 130), text_color, -1)
            cv2.rectangle(frame, (50, 110), (350, 130), text_color, 2)

        else:
            # NO HAND - GRAY
            cv2.putText(frame, 'No hand detected', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Confidence: {(1-confidence/100)*100:.1f}% no hand', (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1, cv2.LINE_AA)

        # Show frame counter
        frame_count += 1
        cv2.putText(frame, f'Frame: {frame_count}', (frame.shape[1]-150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display
        cv2.imshow('Hand Detection - Live Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n👋 Goodbye! Processed {frame_count} frames")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live hand detection demo')
    parser.add_argument('--weights', type=str, default='models/hand_detector.pt',
                       help='Path to trained weights')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Image size for model (default: 224)')
    args = parser.parse_args()

    main(args.weights, args.imgsz)