#!/usr/bin/env python3
"""
Capture ONLY arm images to add to existing hand_cls dataset
Uses clean frame saving like ml-visions (no overlays in saved images)
20 seconds at 10 fps = 200 frames
"""

import cv2
import os
import time
from pathlib import Path
import shutil
from datetime import datetime

class ArmCapture:
    def __init__(self):
        self.temp_dir = Path("temp_arm_captures")
        self.data_dir = Path("data/hand_cls")  # Use existing hand_cls structure
        self.fps = 10  # 10 frames per second
        self.duration = 20  # 20 seconds
        self.total_frames = self.fps * self.duration  # 200 frames

    def capture_arms(self):
        """Capture arm images with clean frames (no overlays)"""
        print("\n📸 ARM Capture Mode")
        print("=" * 50)
        print("💪 Instructions: Show your ARM/FOREARM (not focused on hand)")
        print("   - Show forearm, elbow area")
        print("   - Keep hand out of focus or partially visible")
        print("   - Move arm naturally")
        print(f"\nCapturing {self.total_frames} frames ({self.duration} seconds at {self.fps} fps)")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return None

        # Create temp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_session = self.temp_dir / timestamp
        temp_session.mkdir(parents=True, exist_ok=True)

        # 3-second countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                display_frame = cv2.flip(frame, 1)  # Mirror for display
                cv2.putText(display_frame, f"GET READY: ARM CAPTURE", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(display_frame, str(i), (320, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 165, 255), 3)
                cv2.putText(display_frame, "Show ARM/FOREARM (not hand)", (50, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow('Arm Capture', display_frame)
                cv2.waitKey(1000)

        # Capture frames
        frames_captured = 0
        start_time = time.time()
        frame_interval = 1.0 / self.fps
        last_capture_time = 0

        print("🔴 RECORDING...")

        while frames_captured < self.total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timing
            current_time = time.time() - start_time
            remaining = self.duration - (time.time() - start_time)

            # Save CLEAN frame at specified FPS (no overlays)
            if current_time - last_capture_time >= frame_interval:
                # Save the ORIGINAL frame without any overlays
                clean_frame = cv2.flip(frame, 1)  # Just mirror, no text
                filename = temp_session / f"arm_{frames_captured:04d}.jpg"
                cv2.imwrite(str(filename), clean_frame)
                frames_captured += 1
                last_capture_time = current_time

            # Create display frame with overlays (for user feedback only)
            display_frame = cv2.flip(frame, 1)

            # Progress bar
            progress = frames_captured / self.total_frames
            bar_width = int(progress * 400)
            cv2.rectangle(display_frame, (50, 30), (450, 60), (200, 200, 200), 2)
            cv2.rectangle(display_frame, (50, 30), (50 + bar_width, 60), (0, 165, 255), -1)

            # Text overlays (only on display, not saved)
            cv2.putText(display_frame, "ARM CAPTURE", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            cv2.putText(display_frame, f"Frames: {frames_captured}/{self.total_frames}", (50, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {remaining:.1f}s", (50, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Arm Capture', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Capture interrupted")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Captured {frames_captured} clean frames (no overlays)")
        return temp_session, frames_captured

    def review_and_add(self, temp_session, count):
        """Review captured images and add to dataset"""
        print(f"\n📊 Review: {count} arm images captured")
        print("\nWhat would you like to do?")
        print("  1. ✅ ADD to training dataset")
        print("  2. 🗑️  DELETE them (discard)")
        print("  3. 👀 VIEW them first")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()

            if choice == '1':
                # Add to unified dataset
                self.add_to_unified(temp_session)
                print(f"✅ Added {count} arm images to unified dataset")
                shutil.rmtree(temp_session)
                return True

            elif choice == '2':
                shutil.rmtree(temp_session)
                print(f"🗑️  Deleted {count} temporary images")
                return False

            elif choice == '3':
                import subprocess
                import platform
                if platform.system() == 'Darwin':
                    subprocess.run(['open', str(temp_session)])
                print(f"\n📂 Opened {temp_session}")
                print("Review the images, then choose 1 or 2")

    def add_to_unified(self, temp_session):
        """Add arm images to unified dataset"""
        images = sorted(list(temp_session.glob("*.jpg")))

        # 80/20 split
        train_count = int(len(images) * 0.8)
        train_images = images[:train_count]
        val_images = images[train_count:]

        # Get next indices
        train_dir = self.data_dir / 'train' / 'arm'
        val_dir = self.data_dir / 'val' / 'arm'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_existing = len(list(train_dir.glob("*.jpg")))
        val_existing = len(list(val_dir.glob("*.jpg")))

        # Copy to train
        for i, img in enumerate(train_images):
            dest = train_dir / f"arm_{train_existing + i:04d}.jpg"
            shutil.copy2(img, dest)

        # Copy to val
        for i, img in enumerate(val_images):
            dest = val_dir / f"arm_{val_existing + i:04d}.jpg"
            shutil.copy2(img, dest)

        print(f"  Added {len(train_images)} to training")
        print(f"  Added {len(val_images)} to validation")

    def show_stats(self):
        """Show current hand_cls dataset statistics"""
        print("\n📊 Dataset Statistics (hand_cls):")
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

        print(f"\nTOTAL: {total} images")
        return total

    def run(self):
        """Main capture workflow"""
        print("\n🎯 Arm Capture Tool")
        print("=" * 50)
        print("This adds ARM images to your existing hand/not_hand dataset")

        # Show current stats
        self.show_stats()

        print("\n" + "=" * 50)
        input("Press ENTER when ready to capture ARM images...")

        # Capture arms
        result = self.capture_arms()
        if result:
            temp_session, count = result
            if self.review_and_add(temp_session, count):
                # Show final stats
                total = self.show_stats()
                print("\n🚀 Ready to train unified model!")
                print("Next: python3 train_unified.py")

        # Clean up
        if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
            self.temp_dir.rmdir()

if __name__ == "__main__":
    capture = ArmCapture()
    capture.run()