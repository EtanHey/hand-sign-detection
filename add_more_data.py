#!/usr/bin/env python3
"""
Interactive data collection to ADD to existing dataset.
Capture more images to improve model accuracy.
"""

import cv2
import os
from pathlib import Path
from datetime import datetime
import shutil

class DatasetManager:
    def __init__(self):
        self.data_root = Path("data")
        self.hand_cls_dir = self.data_root / "hand_cls"
        self.raw_dir = self.data_root / "raw"

    def show_current_stats(self):
        """Show current dataset statistics"""
        print("\n📊 Current Dataset Statistics:")
        print("-" * 50)

        # Check hand_cls directory (from ml-visions)
        if self.hand_cls_dir.exists():
            train_hand = len(list((self.hand_cls_dir / "train/hand").glob("*.*"))) if (self.hand_cls_dir / "train/hand").exists() else 0
            train_not = len(list((self.hand_cls_dir / "train/not_hand").glob("*.*"))) if (self.hand_cls_dir / "train/not_hand").exists() else 0
            val_hand = len(list((self.hand_cls_dir / "val/hand").glob("*.*"))) if (self.hand_cls_dir / "val/hand").exists() else 0
            val_not = len(list((self.hand_cls_dir / "val/not_hand").glob("*.*"))) if (self.hand_cls_dir / "val/not_hand").exists() else 0

            print(f"📁 hand_cls/ (training data):")
            print(f"   Training:   {train_hand:4d} hands, {train_not:4d} not_hands")
            print(f"   Validation: {val_hand:4d} hands, {val_not:4d} not_hands")
            print(f"   Total:      {train_hand + train_not + val_hand + val_not:4d} images")

        # Check raw directory (new captures)
        if self.raw_dir.exists():
            raw_hand = len(list((self.raw_dir / "hand").glob("*.jpg"))) if (self.raw_dir / "hand").exists() else 0
            raw_not = len(list((self.raw_dir / "not_hand").glob("*.jpg"))) if (self.raw_dir / "not_hand").exists() else 0

            if raw_hand > 0 or raw_not > 0:
                print(f"\n📁 raw/ (new captures):")
                print(f"   Hands:      {raw_hand:4d} images")
                print(f"   Not hands:  {raw_not:4d} images")

        print("-" * 50)

    def capture_images(self, category="hand", count=50):
        """Capture images to TEMP folder first"""
        from datetime import datetime

        # Create temp directory for this session
        temp_session = Path("temp_captures") / f"{category}_{datetime.now().strftime('%H%M%S')}"
        temp_session.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Capturing {count} images for '{category}'")
        print(f"   Temporary location: {temp_session}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return 0

        print("\nControls:")
        print("  SPACE - Capture image")
        print("  S - Start/stop auto-capture (3 fps)")
        print("  Q - Finish and return to menu")

        if category == "hand":
            print("\n✋ Show your hands in different positions!")
        else:
            print("\n🚫 Keep hands OUT of frame!")

        captured = 0
        auto_capture = False
        last_capture_time = 0

        while captured < count:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for better UX
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # Status overlay
            status_color = (0, 255, 0) if category == "hand" else (0, 0, 255)
            cv2.putText(display_frame, f"Capturing: {category}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"Progress: {captured}/{count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if auto_capture:
                cv2.putText(display_frame, "AUTO-CAPTURE ON", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Progress bar
            bar_width = int((captured / count) * 400)
            cv2.rectangle(display_frame, (10, 100), (410, 120), (100, 100, 100), 2)
            cv2.rectangle(display_frame, (10, 100), (10 + bar_width, 120), status_color, -1)

            cv2.imshow('Data Collection', display_frame)

            # Auto-capture mode
            import time
            current_time = time.time()
            if auto_capture and (current_time - last_capture_time) >= 0.33:  # ~3 fps
                filename = f"{category}_{captured:04d}.jpg"
                filepath = temp_session / filename
                cv2.imwrite(str(filepath), frame)
                captured += 1
                last_capture_time = current_time
                print(f"  Captured: {filename}")

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and not auto_capture:
                # Manual capture
                filename = f"{category}_{captured:04d}.jpg"
                filepath = temp_session / filename
                cv2.imwrite(str(filepath), frame)
                captured += 1
                print(f"  Captured: {filename}")
            elif key == ord('s'):
                auto_capture = not auto_capture
                if auto_capture:
                    print("  🔴 Auto-capture ON")
                else:
                    print("  ⏸️  Auto-capture OFF")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Captured {captured} images to temp folder")

        # Ask what to do with the images
        return self.review_captures(temp_session, category, captured)

    def capture_alternating(self, images_per_category=25, switch_interval=5):
        """Capture images alternating between hand and not_hand"""

        print(f"\n🔄 ALTERNATING CAPTURE MODE")
        print(f"   Will capture {images_per_category} images of each type")
        print(f"   Switching every {switch_interval} seconds")
        print("\nControls:")
        print("  Q - Stop and return to menu")
        print("  SPACE - Pause/Resume")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return 0

        # Prepare output directories
        hand_dir = self.hand_cls_dir / "train" / "hand"
        not_hand_dir = self.hand_cls_dir / "train" / "not_hand"
        hand_dir.mkdir(parents=True, exist_ok=True)
        not_hand_dir.mkdir(parents=True, exist_ok=True)

        # Count existing for numbering
        hand_start = len(list(hand_dir.glob("*.*"))) + 1
        not_hand_start = len(list(not_hand_dir.glob("*.*"))) + 1

        hand_captured = 0
        not_hand_captured = 0
        current_category = "hand"

        import time
        last_switch_time = time.time()
        last_capture_time = 0
        capture_fps = 3  # Captures per second
        paused = False

        # Countdown before starting
        print("\n🎬 Starting in 3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print("   GO!\n")

        while hand_captured < images_per_category or not_hand_captured < images_per_category:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            current_time = time.time()

            # Check if we should switch categories
            if not paused and (current_time - last_switch_time) >= switch_interval:
                # Switch category
                if current_category == "hand" and not_hand_captured < images_per_category:
                    current_category = "not_hand"
                    print(f"\n🔄 SWITCH! Now capturing: NOT_HAND")
                    print("   Remove hands from view!")
                elif current_category == "not_hand" and hand_captured < images_per_category:
                    current_category = "hand"
                    print(f"\n🔄 SWITCH! Now capturing: HAND")
                    print("   Show your hands!")
                last_switch_time = current_time

            # Skip if we've captured enough of this category
            if current_category == "hand" and hand_captured >= images_per_category:
                if not_hand_captured < images_per_category:
                    current_category = "not_hand"
                    last_switch_time = current_time
            elif current_category == "not_hand" and not_hand_captured >= images_per_category:
                if hand_captured < images_per_category:
                    current_category = "hand"
                    last_switch_time = current_time

            # Time until next switch
            time_until_switch = switch_interval - (current_time - last_switch_time)

            # Display status
            status_color = (0, 255, 0) if current_category == "hand" else (0, 0, 255)

            # Big instruction
            instruction = "✋ SHOW HANDS!" if current_category == "hand" else "🚫 HIDE HANDS!"
            cv2.putText(display_frame, instruction, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

            # Timer until switch
            cv2.putText(display_frame, f"Switch in: {time_until_switch:.1f}s", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Progress
            cv2.putText(display_frame, f"Hands: {hand_captured}/{images_per_category}", (50, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Not hands: {not_hand_captured}/{images_per_category}", (50, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if paused:
                cv2.putText(display_frame, "PAUSED", (250, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Progress bars
            total_captured = hand_captured + not_hand_captured
            total_needed = images_per_category * 2
            overall_progress = total_captured / total_needed if total_needed > 0 else 0

            bar_width = int(overall_progress * 400)
            cv2.rectangle(display_frame, (50, 200), (450, 220), (100, 100, 100), 2)
            cv2.rectangle(display_frame, (50, 200), (50 + bar_width, 220), (255, 255, 0), -1)

            cv2.imshow('Alternating Data Collection', display_frame)

            # Auto-capture at specified FPS
            if not paused and (current_time - last_capture_time) >= (1.0 / capture_fps):
                if current_category == "hand" and hand_captured < images_per_category:
                    filename = f"hand_{hand_start + hand_captured:04d}.jpg"
                    filepath = hand_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    hand_captured += 1
                    print(f"  📸 Captured: {filename}")
                elif current_category == "not_hand" and not_hand_captured < images_per_category:
                    filename = f"not_hand_{not_hand_start + not_hand_captured:04d}.jpg"
                    filepath = not_hand_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    not_hand_captured += 1
                    print(f"  📸 Captured: {filename}")

                last_capture_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print("  ⏸️  PAUSED")
                else:
                    print("  ▶️  RESUMED")
                    last_switch_time = current_time  # Reset switch timer

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Alternating capture complete!")
        print(f"   Captured {hand_captured} hand images")
        print(f"   Captured {not_hand_captured} not_hand images")
        return hand_captured + not_hand_captured

    def review_captures(self, temp_session, category, count):
        """Ask user what to do with captured images"""
        print("\n" + "="*50)
        print(f"📋 Review {count} captured {category} images")
        print("="*50)

        print("\nWhat would you like to do with these images?")
        print("  1. ✅ ADD to training dataset")
        print("  2. 🗑️  DELETE them (discard)")
        print("  3. 👀 VIEW them first")

        while True:
            choice = input("\nSelect (1-3): ").strip()

            if choice == '1':
                # Add to dataset
                added = self.add_temp_to_dataset(temp_session, category)
                return added

            elif choice == '2':
                # Delete
                import shutil
                shutil.rmtree(temp_session)
                print("🗑️  Images deleted")
                return 0

            elif choice == '3':
                # View images
                self.view_temp_images(temp_session)
                # Ask again after viewing
                continue
            else:
                print("❌ Invalid choice")

    def add_temp_to_dataset(self, temp_session, category):
        """Move images from temp to dataset"""
        target_dir = self.hand_cls_dir / "train" / category
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get starting index
        existing = len(list(target_dir.glob("*.*")))

        # Move files
        moved = 0
        for img_file in temp_session.glob("*.jpg"):
            new_name = f"{category}_{existing + moved + 1:04d}.jpg"
            dest = target_dir / new_name
            import shutil
            shutil.move(str(img_file), str(dest))
            moved += 1

        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_session)

        print(f"✅ Added {moved} images to dataset")
        return moved

    def view_temp_images(self, temp_session):
        """Quick view of captured images"""
        images = list(temp_session.glob("*.jpg"))
        print(f"\n👀 Viewing {len(images)} images (Press SPACE for next, Q to stop)")

        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            cv2.putText(img, f"Image {i+1}/{len(images)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Review', img)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def merge_raw_to_training(self):
        """Merge raw captured images into training set"""
        if not self.raw_dir.exists():
            print("❌ No raw directory found")
            return

        moved = 0
        for category in ["hand", "not_hand"]:
            raw_cat_dir = self.raw_dir / category
            if not raw_cat_dir.exists():
                continue

            train_cat_dir = self.hand_cls_dir / "train" / category
            train_cat_dir.mkdir(parents=True, exist_ok=True)

            # Get starting index
            existing = len(list(train_cat_dir.glob("*.*")))

            # Move files
            for i, img_file in enumerate(raw_cat_dir.glob("*.jpg")):
                new_name = f"{category}_{existing + i + 1:04d}.jpg"
                dest = train_cat_dir / new_name
                shutil.move(str(img_file), str(dest))
                moved += 1

        if moved > 0:
            print(f"✅ Moved {moved} images from raw/ to hand_cls/train/")
            # Clean up empty directories
            try:
                shutil.rmtree(self.raw_dir)
            except:
                pass
        else:
            print("❌ No images to move")

def main():
    manager = DatasetManager()

    while True:
        print("\n" + "="*60)
        print("🖐️  Hand Detection - Add More Training Data")
        print("="*60)

        manager.show_current_stats()

        print("\nOptions:")
        print("  1. Add more HAND images")
        print("  2. Add more NOT_HAND images")
        print("  3. Quick capture (25 hands + 25 not_hands)")
        print("  4. 🔄 ALTERNATING capture (auto-switches every 5 seconds)")
        print("  5. Merge raw/ images into training set")
        print("  6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            count = input("How many hand images to capture? [50]: ").strip()
            count = int(count) if count else 50
            manager.capture_images("hand", count)

        elif choice == '2':
            count = input("How many not_hand images to capture? [50]: ").strip()
            count = int(count) if count else 50
            manager.capture_images("not_hand", count)

        elif choice == '3':
            print("\n🚀 Quick capture mode")
            manager.capture_images("hand", 25)
            print("\n💨 Switch - remove hands from view!")
            import time
            time.sleep(3)
            manager.capture_images("not_hand", 25)

        elif choice == '4':
            print("\n🔄 ALTERNATING CAPTURE MODE")
            count = input("How many images per category? [25]: ").strip()
            count = int(count) if count else 25
            interval = input("Seconds between switches? [5]: ").strip()
            interval = int(interval) if interval else 5
            manager.capture_alternating(count, interval)

        elif choice == '5':
            manager.merge_raw_to_training()

        elif choice == '6':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")

    print("\n✅ Data collection complete!")
    print("\n🚀 Next steps:")
    print("   1. Review your images in data/hand_cls/")
    print("   2. Retrain: python3 train_hands_only.py")
    print("   3. Test: python3 live_demo.py")

if __name__ == "__main__":
    main()