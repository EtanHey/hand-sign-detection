#!/usr/bin/env python3
"""
Interactive hand detection tester
Test your trained model with webcam or images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from PIL import Image

class HandTester:
    def __init__(self, model_path="models/hand_detector.pt"):
        """Initialize with trained model"""
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print("   Train first with: python3 train_hands_only.py")
            exit(1)

        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Get class names
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = {0: 'hand', 1: 'not_hand'}

        print(f"✅ Model loaded! Classes: {list(self.class_names.values())}")

    def test_image(self, image_path):
        """Test a single image"""
        print(f"\n🖼️  Testing image: {image_path}")

        # Run prediction
        results = self.model(image_path)

        if results and len(results) > 0:
            result = results[0]

            # For classification model
            if hasattr(result, 'probs'):
                probs = result.probs
                top1_idx = probs.top1
                top1_conf = probs.top1conf.item()

                class_name = self.class_names.get(top1_idx, f"Class {top1_idx}")

                print(f"\n📊 Result:")
                print(f"   Prediction: {class_name}")
                print(f"   Confidence: {top1_conf:.2%}")

                # Show confidence for all classes
                if probs.data is not None:
                    print(f"\n   All probabilities:")
                    for idx, conf in enumerate(probs.data):
                        name = self.class_names.get(idx, f"Class {idx}")
                        print(f"   - {name}: {conf:.2%}")

                # Display result
                if class_name == 'hand':
                    print("\n✋ Hand detected!")
                else:
                    print("\n❌ No hand detected")

                return class_name, top1_conf

        return None, 0

    def test_webcam(self):
        """Live webcam testing"""
        print("\n📹 Starting webcam test...")
        print("   Press 'q' to quit")
        print("   Press SPACE to capture and analyze frame")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for better UX
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # Add instructions
            cv2.putText(display_frame, "Press SPACE to test | Q to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Hand Detection Test", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # Save frame temporarily
                temp_path = "temp_test.jpg"
                cv2.imwrite(temp_path, frame)

                # Test it
                print("\n" + "="*50)
                class_name, confidence = self.test_image(temp_path)

                # Show result on frame for 2 seconds
                result_frame = frame.copy()
                color = (0, 255, 0) if class_name == 'hand' else (0, 0, 255)
                text = f"{class_name}: {confidence:.1%}"

                # Draw result with background
                cv2.rectangle(result_frame, (10, 50), (300, 100), color, -1)
                cv2.putText(result_frame, text, (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Hand Detection Test", result_frame)
                cv2.waitKey(2000)  # Show for 2 seconds

                # Clean up temp file
                Path(temp_path).unlink()

        cap.release()
        cv2.destroyAllWindows()

    def test_batch(self, folder_path):
        """Test all images in a folder"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return

        print(f"\n📁 Testing {len(image_files)} images from {folder_path}")

        results = []
        for img_path in image_files[:10]:  # Test first 10
            class_name, conf = self.test_image(img_path)
            results.append((img_path.name, class_name, conf))

        # Summary
        print("\n" + "="*50)
        print("📊 Summary:")
        correct_hands = sum(1 for _, c, _ in results if c == 'hand' and 'hand' in _.lower())
        correct_not = sum(1 for _, c, _ in results if c == 'not_hand' and 'not' in _.lower())

        print(f"   Tested: {len(results)} images")
        if correct_hands + correct_not > 0:
            accuracy = (correct_hands + correct_not) / len(results)
            print(f"   Approximate accuracy: {accuracy:.1%}")

def main():
    parser = argparse.ArgumentParser(description="Test hand detection model")
    parser.add_argument('mode', choices=['webcam', 'image', 'folder'],
                       help='Test mode')
    parser.add_argument('path', nargs='?',
                       help='Path to image or folder (not needed for webcam)')
    parser.add_argument('--model', default='models/hand_detector.pt',
                       help='Path to model')

    args = parser.parse_args()

    # Initialize tester
    tester = HandTester(args.model)

    if args.mode == 'webcam':
        tester.test_webcam()
    elif args.mode == 'image':
        if not args.path:
            print("❌ Please provide image path")
            exit(1)
        tester.test_image(args.path)
    elif args.mode == 'folder':
        if not args.path:
            # Default to validation folder
            args.path = "data/hand_cls/val/hand"
        tester.test_batch(args.path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("\n🖐️  Hand Detection Tester")
        print("="*40)
        print("\nUsage:")
        print("  1. Test with webcam:")
        print("     python3 test_hands.py webcam")
        print("\n  2. Test single image:")
        print("     python3 test_hands.py image photo.jpg")
        print("\n  3. Test folder of images:")
        print("     python3 test_hands.py folder data/hand_cls/val/hand")
        print("\nChoose option (1/2/3): ", end="")

        choice = input().strip()

        if choice == '1':
            sys.argv = ['test_hands.py', 'webcam']
        elif choice == '2':
            print("Enter image path: ", end="")
            path = input().strip()
            sys.argv = ['test_hands.py', 'image', path]
        elif choice == '3':
            print("Enter folder path (or press Enter for default): ", end="")
            path = input().strip() or "data/hand_cls/val/hand"
            sys.argv = ['test_hands.py', 'folder', path]
        else:
            print("Invalid choice")
            exit(1)

    main()