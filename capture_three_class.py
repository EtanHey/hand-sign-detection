#!/usr/bin/env python3
"""
Three-class data capture for hand/arm/not_hand detection
Captures 20 seconds at 10 fps = 200 frames per class
"""

import cv2
import os
import time
from pathlib import Path
import shutil
from datetime import datetime

class ThreeClassCapture:
    def __init__(self):
        self.temp_dir = Path("temp_captures")
        self.data_dir = Path("data/three_class")
        self.fps = 10  # 10 frames per second
        self.duration = 20  # 20 seconds per class
        self.total_frames = self.fps * self.duration  # 200 frames

    def setup_directories(self):
        """Create necessary directories"""
        # Create temp directories
        for category in ['hand', 'arm', 'not_hand']:
            (self.temp_dir / category).mkdir(parents=True, exist_ok=True)

        # Create data directories
        for split in ['train', 'val']:
            for category in ['hand', 'arm', 'not_hand']:
                (self.data_dir / split / category).mkdir(parents=True, exist_ok=True)

    def review_captures(self, temp_session, class_name, count):
        """Review captured images and decide what to do"""
        print(f"\n📊 Review: {count} {class_name} images captured")
        print("\nWhat would you like to do with these images?")
        print("  1. ✅ ADD to training dataset")
        print("  2. 🗑️  DELETE them (discard)")
        print("  3. 👀 VIEW them first (opens folder)")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()

            if choice == '1':
                # Add to dataset
                self.add_to_dataset(temp_session, class_name)
                print(f"✅ Added {count} images to {class_name} dataset")
                return True

            elif choice == '2':
                # Delete
                shutil.rmtree(temp_session)
                print(f"🗑️  Deleted {count} temporary images")
                return False

            elif choice == '3':
                # Open folder for viewing
                import subprocess
                import platform
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', str(temp_session)])
                elif platform.system() == 'Windows':
                    subprocess.run(['explorer', str(temp_session)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(temp_session)])
                print(f"\n📂 Opened {temp_session}")
                print("Review the images, then choose:")
                print("  1. ✅ ADD to dataset")
                print("  2. 🗑️  DELETE them")
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def add_to_dataset(self, temp_session, class_name):
        """Add captured images to the dataset with 80/20 split"""
        images = sorted(list(temp_session.glob("*.jpg")))

        # 80/20 train/val split
        train_count = int(len(images) * 0.8)
        train_images = images[:train_count]
        val_images = images[train_count:]

        # Get next available indices
        train_dir = self.data_dir / 'train' / class_name
        val_dir = self.data_dir / 'val' / class_name

        train_existing = len(list(train_dir.glob("*.jpg")))
        val_existing = len(list(val_dir.glob("*.jpg")))

        # Copy to train
        for i, img in enumerate(train_images):
            dest = train_dir / f"{class_name}_{train_existing + i:04d}.jpg"
            shutil.copy2(img, dest)

        # Copy to val
        for i, img in enumerate(val_images):
            dest = val_dir / f"{class_name}_{val_existing + i:04d}.jpg"
            shutil.copy2(img, dest)

        # Clean up temp
        shutil.rmtree(temp_session)

        print(f"  Added {len(train_images)} to training")
        print(f"  Added {len(val_images)} to validation")

    def show_dataset_stats(self):
        """Show current dataset statistics"""
        print("\n📊 Current Dataset Statistics:")
        print("=" * 50)

        total = 0
        for split in ['train', 'val']:
            print(f"\n{split.upper()}:")
            for category in ['hand', 'arm', 'not_hand']:
                path = self.data_dir / split / category
                if path.exists():
                    count = len(list(path.glob("*.jpg")))
                    total += count
                    print(f"  {category:10s}: {count:4d} images")
                else:
                    print(f"  {category:10s}:    0 images")

        print(f"\nTOTAL: {total} images")
        return total

    def capture_all_classes(self):
        """Capture all three classes in one continuous session"""
        print("\n📸 Starting continuous capture session")
        print("=" * 50)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False

        # Create timestamp for this capture session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Classes to capture
        classes = [
            ('hand', '✋ Show your HAND clearly (palm, fist, fingers, etc.)'),
            ('arm', '💪 Show your ARM/FOREARM (not focused on hand)'),
            ('not_hand', '🚫 Show NO hands or arms (face, objects, background)')
        ]

        captured_frames = {}

        for class_idx, (class_name, instructions) in enumerate(classes):
            print(f"\n📸 {class_idx + 1}/3: {class_name.upper()}")
            print(instructions)

            # Create temp directory for this class
            temp_session = self.temp_dir / class_name / timestamp
            temp_session.mkdir(parents=True, exist_ok=True)
            captured_frames[class_name] = {'path': temp_session, 'count': 0}

            # 3-second countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"GET READY: {class_name.upper()}", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(frame, str(i), (320, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 3)
                    cv2.putText(frame, instructions, (50, 400),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.imshow('Three-Class Capture', frame)
                    cv2.waitKey(1000)

            # Capture for this class
            frames_captured = 0
            start_time = time.time()
            frame_interval = 1.0 / self.fps
            last_capture_time = 0

            print("🔴 RECORDING...")

            while frames_captured < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror for better UX
                frame = cv2.flip(frame, 1)

                # Calculate timing
                elapsed = time.time() - start_time
                remaining = self.duration - elapsed

                # Color coding for each class
                if class_name == 'hand':
                    color = (0, 0, 255)  # Red
                elif class_name == 'arm':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (128, 128, 128)  # Gray

                # Progress bar
                progress = frames_captured / self.total_frames
                bar_width = int(progress * 400)
                cv2.rectangle(frame, (50, 30), (450, 60), (200, 200, 200), 2)
                cv2.rectangle(frame, (50, 30), (50 + bar_width, 60), color, -1)

                # Text overlays
                cv2.putText(frame, f"{class_idx + 1}/3: {class_name.upper()}", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Frames: {frames_captured}/{self.total_frames}", (50, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {remaining:.1f}s", (50, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Instructions reminder
                cv2.putText(frame, instructions, (50, 420),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow('Three-Class Capture', frame)

                # Capture at specified FPS
                current_time = time.time() - start_time
                if current_time - last_capture_time >= frame_interval:
                    # Save frame
                    filename = temp_session / f"{class_name}_{frames_captured:04d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    frames_captured += 1
                    last_capture_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Capture interrupted")
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured_frames

            captured_frames[class_name]['count'] = frames_captured
            print(f"✅ Captured {frames_captured} frames for {class_name}")

        cap.release()
        cv2.destroyAllWindows()

        return captured_frames

    def run(self):
        """Main capture workflow"""
        print("\n🎯 Three-Class Hand/Arm Detection Data Capture")
        print("=" * 50)
        print("This tool will capture:")
        print(f"  • 20 seconds per class at {self.fps} fps")
        print(f"  • {self.total_frames} frames per class")
        print("  • 3 classes: hand, arm, not_hand")
        print(f"  • Total: {self.total_frames * 3} frames in one session")
        print("\n⚡ Camera stays open for entire capture!")

        self.setup_directories()

        # Show current stats
        self.show_dataset_stats()

        print("\n" + "=" * 50)
        input("Press ENTER when ready to start continuous capture...")

        # Capture all classes in one session
        captured_frames = self.capture_all_classes()

        if not captured_frames:
            print("\n❌ Capture failed or was interrupted")
            return

        # Review all captures at once
        print("\n" + "=" * 50)
        print("📊 CAPTURE COMPLETE!")
        print("=" * 50)

        total_captured = sum(info['count'] for info in captured_frames.values())
        print(f"\nCaptured {total_captured} total frames:")
        for class_name, info in captured_frames.items():
            print(f"  • {class_name}: {info['count']} frames")

        print("\nWhat would you like to do with ALL captured images?")
        print("  1. ✅ ADD all to training dataset")
        print("  2. 🗑️  DELETE all (discard)")
        print("  3. 👀 VIEW them first (opens folders)")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()

            if choice == '1':
                # Add all to dataset
                for class_name, info in captured_frames.items():
                    if info['count'] > 0:
                        self.add_to_dataset(info['path'], class_name)
                        print(f"✅ Added {info['count']} {class_name} images")
                break

            elif choice == '2':
                # Delete all
                for class_name, info in captured_frames.items():
                    if info['path'].exists():
                        shutil.rmtree(info['path'])
                print(f"🗑️  Deleted all temporary images")
                break

            elif choice == '3':
                # Open folders for viewing
                import subprocess
                import platform
                for class_name, info in captured_frames.items():
                    if info['path'].exists():
                        if platform.system() == 'Darwin':  # macOS
                            subprocess.run(['open', str(info['path'])])
                        elif platform.system() == 'Windows':
                            subprocess.run(['explorer', str(info['path'])])
                        else:  # Linux
                            subprocess.run(['xdg-open', str(info['path'])])
                print(f"\n📂 Opened all capture folders for review")
                print("Review the images, then choose:")
                print("  1. ✅ ADD all to dataset")
                print("  2. 🗑️  DELETE all")
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        # Final stats
        print("\n" + "=" * 50)
        print("🎉 SESSION COMPLETE!")
        total = self.show_dataset_stats()

        if total > 0:
            print("\n🚀 Ready to train!")
            print("Next step: python3 train_three_class.py")

        # Clean up temp directory if empty
        if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
            self.temp_dir.rmdir()

if __name__ == "__main__":
    capture = ThreeClassCapture()
    capture.run()