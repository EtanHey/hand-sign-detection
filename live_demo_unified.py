#!/usr/bin/env python3
"""
Live demo using the unified three-class model
Shows real-time detection of hand/arm/not_hand
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def main():
    # Load the latest unified model
    model_path = Path("models/unified_detector.pt")

    if not model_path.exists():
        # Fallback to versioned model
        model_path = Path("models/unified_v2.pt")

    if not model_path.exists():
        print("❌ No unified model found! Train one first with:")
        print("   python3 train_unified.py")
        return

    print(f"✅ Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    print("\n🎥 Live Demo Started!")
    print("Press 'q' to quit")
    print("\nModel distinguishes:")
    print("  • HAND: Close-up hand with fingers visible")
    print("  • ARM: Forearm or elbow area")
    print("  • NOT_HAND: Neither hand nor arm\n")

    # Colors for different classes
    colors = {
        'hand': (0, 255, 0),      # Green
        'arm': (0, 165, 255),      # Orange
        'not_hand': (0, 0, 255)    # Red
    }

    # Class names (YOLO uses alphabetical order!)
    classes = ['arm', 'hand', 'not_hand']  # Index 0=arm, 1=hand, 2=not_hand

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Run inference
        results = model(frame, verbose=False)

        if results and results[0].probs is not None:
            probs = results[0].probs

            # Get top prediction
            top_class_idx = probs.top1
            top_confidence = float(probs.top1conf)
            class_name = classes[top_class_idx]

            # Get color based on class
            color = colors[class_name]

            # Draw thick border around frame based on detection
            thickness = 5
            cv2.rectangle(frame, (thickness, thickness),
                         (frame.shape[1] - thickness, frame.shape[0] - thickness),
                         color, thickness)

            # Prepare status text
            if class_name == 'hand':
                status = f"✋ HAND: {top_confidence:.1%}"
                emoji = "✋"
            elif class_name == 'arm':
                status = f"💪 ARM: {top_confidence:.1%}"
                emoji = "💪"
            else:
                status = f"❌ NO HAND/ARM: {top_confidence:.1%}"
                emoji = "❌"

            # Draw status background
            cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 90), color, 2)

            # Draw main status
            cv2.putText(frame, status, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw all probabilities
            y_offset = 70
            for i, cls in enumerate(classes):
                prob = float(probs.data[i])
                text = f"{cls}: {prob:.1%}"
                text_color = (150, 150, 150) if i != top_class_idx else color
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                # Draw probability bar
                bar_width = int(prob * 100)
                bar_color = colors[cls] if i == top_class_idx else (100, 100, 100)
                cv2.rectangle(frame, (130, y_offset - 12),
                             (130 + bar_width, y_offset - 2),
                             bar_color, -1)
                y_offset += 20

            # Instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("Unified Hand/Arm Detection", frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Demo ended")

if __name__ == "__main__":
    main()