#!/usr/bin/env python3
"""
Hand Data Collection Tool
Captures images from webcam for training hand detection and gesture recognition models
"""

import cv2
import os
import time
from datetime import datetime
from pathlib import Path
import argparse

class HandDataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Gesture categories we want to collect
        self.gestures = {
            'hand': 'Open hand (palm facing camera)',
            'ok': 'OK sign 👌',
            'thumbs_up': 'Thumbs up 👍',
            'peace': 'Peace sign ✌️',
            'fist': 'Closed fist',
            'point': 'Pointing finger',
            'rock': 'Rock sign 🤘',
            'wave': 'Waving hand',
            'stop': 'Stop gesture (palm out)',
            'none': 'No hand (background)'
        }

    def create_session_folder(self, gesture_name):
        """Create folder for current collection session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / gesture_name / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def collect_gesture(self, gesture_name):
        """Collect images for a specific gesture"""
        if gesture_name not in self.gestures:
            print(f"Unknown gesture: {gesture_name}")
            return

        session_dir = self.create_session_folder(gesture_name)
        cap = cv2.VideoCapture(0)

        # Camera settings for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n📸 Collecting data for: {gesture_name}")
        print(f"   Description: {self.gestures[gesture_name]}")
        print(f"   Saving to: {session_dir}")
        print("\nControls:")
        print("  SPACE - Capture image")
        print("  S - Start/stop continuous capture (3 fps)")
        print("  Q - Quit and return to menu")
        print("\n💡 Tips: Vary hand position, angle, distance, and lighting\n")

        image_count = 0
        continuous_capture = False
        last_capture_time = 0
        capture_interval = 0.33  # ~3 fps for continuous mode

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the image for more intuitive experience
            frame = cv2.flip(frame, 1)

            # Display info on frame
            info_text = f"Gesture: {gesture_name} | Images: {image_count}"
            if continuous_capture:
                info_text += " | RECORDING"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw center crosshair for hand placement
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2-50, h//2), (w//2+50, h//2), (128, 128, 128), 1)
            cv2.line(frame, (w//2, h//2-50), (w//2, h//2+50), (128, 128, 128), 1)

            cv2.imshow('Hand Data Collection', frame)

            # Continuous capture mode
            current_time = time.time()
            if continuous_capture and (current_time - last_capture_time) >= capture_interval:
                filename = f"{gesture_name}_{image_count:04d}.jpg"
                filepath = session_dir / filename
                cv2.imwrite(str(filepath), frame)
                image_count += 1
                last_capture_time = current_time
                print(f"  Captured: {filename}")

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and not continuous_capture:
                # Single capture
                filename = f"{gesture_name}_{image_count:04d}.jpg"
                filepath = session_dir / filename
                cv2.imwrite(str(filepath), frame)
                image_count += 1
                print(f"  Captured: {filename}")
            elif key == ord('s'):
                continuous_capture = not continuous_capture
                if continuous_capture:
                    print("  🔴 Continuous capture ON")
                else:
                    print("  ⏸️  Continuous capture OFF")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Collected {image_count} images for '{gesture_name}'")
        return image_count

    def interactive_collection(self):
        """Interactive menu for data collection"""
        while True:
            print("\n" + "="*50)
            print("Hand Gesture Data Collection")
            print("="*50)
            print("\nAvailable gestures:")
            for i, (key, desc) in enumerate(self.gestures.items(), 1):
                print(f"  {i}. {key:12} - {desc}")

            print("\n  0. Exit")
            print("\nTip: Collect 50-100 images per gesture with varied:")
            print("  - Hand positions and angles")
            print("  - Distances from camera")
            print("  - Lighting conditions")
            print("  - Backgrounds")

            choice = input("\nSelect gesture to collect (0-{}): ".format(len(self.gestures)))

            try:
                choice_idx = int(choice)
                if choice_idx == 0:
                    print("\nGoodbye!")
                    break
                elif 1 <= choice_idx <= len(self.gestures):
                    gesture_name = list(self.gestures.keys())[choice_idx - 1]
                    self.collect_gesture(gesture_name)
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")

    def show_statistics(self):
        """Show collected data statistics"""
        print("\n📊 Data Collection Statistics:")
        print("-" * 40)

        total_images = 0
        for gesture_dir in self.output_dir.iterdir():
            if gesture_dir.is_dir():
                image_count = sum(1 for f in gesture_dir.rglob("*.jpg"))
                if image_count > 0:
                    print(f"  {gesture_dir.name:12} : {image_count:4} images")
                    total_images += image_count

        print("-" * 40)
        print(f"  Total        : {total_images:4} images")

def main():
    parser = argparse.ArgumentParser(description="Collect hand gesture data")
    parser.add_argument('--output', default='data/raw', help='Output directory')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    args = parser.parse_args()

    collector = HandDataCollector(args.output)

    if args.stats:
        collector.show_statistics()
    else:
        try:
            collector.interactive_collection()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            collector.show_statistics()

if __name__ == "__main__":
    main()