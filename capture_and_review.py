#!/usr/bin/env python3
"""
Capture images with review option before adding to dataset.
Images are captured to a temporary folder first.
"""

import cv2
import os
import shutil
from pathlib import Path
from datetime import datetime

class CaptureWithReview:
    def __init__(self):
        self.temp_dir = Path("temp_captures")
        self.data_dir = Path("data/hand_cls")

    def capture_session(self, category="hand", count=50):
        """Capture images to temporary folder"""

        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        session_dir = self.temp_dir / f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Capturing {count} '{category}' images")
        print(f"   Temporary location: {session_dir}")
        print("\nControls:")
        print("  SPACE - Capture image")
        print("  S - Start/stop auto-capture (3 fps)")
        print("  Q - Finish capturing")

        if category == "hand":
            print("\n✋ Show your hands in different positions!")
        else:
            print("\n🚫 Keep hands OUT of frame!")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None

        captured = 0
        auto_capture = False
        last_capture_time = 0

        while captured < count:
            ret, frame = cap.read()
            if not ret:
                break

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

            cv2.imshow('Capture Session', display_frame)

            # Auto-capture
            import time
            current_time = time.time()
            if auto_capture and (current_time - last_capture_time) >= 0.33:
                filename = f"{category}_{captured:04d}.jpg"
                filepath = session_dir / filename
                cv2.imwrite(str(filepath), frame)
                captured += 1
                last_capture_time = current_time
                print(f"  📸 {captured}/{count}")

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and not auto_capture:
                filename = f"{category}_{captured:04d}.jpg"
                filepath = session_dir / filename
                cv2.imwrite(str(filepath), frame)
                captured += 1
                print(f"  📸 {captured}/{count}")
            elif key == ord('s'):
                auto_capture = not auto_capture
                if auto_capture:
                    print("  🔴 Auto-capture ON")
                else:
                    print("  ⏸️  Auto-capture OFF")

        cap.release()
        cv2.destroyAllWindows()

        if captured > 0:
            print(f"\n✅ Captured {captured} images")
            return session_dir, category, captured
        else:
            # Clean up empty directory
            shutil.rmtree(session_dir)
            return None

    def review_session(self, session_info):
        """Review captured images and decide what to do"""
        if not session_info:
            return

        session_dir, category, count = session_info

        while True:
            print("\n" + "="*50)
            print("📋 REVIEW CAPTURED IMAGES")
            print("="*50)
            print(f"\n📁 Session: {session_dir.name}")
            print(f"   Category: {category}")
            print(f"   Images: {count}")

            print("\nWhat would you like to do?")
            print("  1. ✅ KEEP - Add to training dataset")
            print("  2. 👀 VIEW - Look at the images first")
            print("  3. 🗑️  DELETE - Discard these images")
            print("  4. 📁 KEEP IN TEMP - Keep for later review")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                # Add to dataset
                self.add_to_dataset(session_dir, category)
                return

            elif choice == '2':
                # View images
                self.view_images(session_dir)

            elif choice == '3':
                # Delete
                confirm = input("Are you sure you want to DELETE these images? (y/n): ").strip().lower()
                if confirm == 'y':
                    shutil.rmtree(session_dir)
                    print("🗑️  Images deleted")
                    return

            elif choice == '4':
                # Keep in temp
                print(f"📁 Images kept in: {session_dir}")
                print("   You can review them later")
                return
            else:
                print("❌ Invalid choice")

    def add_to_dataset(self, session_dir, category):
        """Move images from temp to training dataset"""
        target_dir = self.data_dir / "train" / category
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get starting index
        existing = len(list(target_dir.glob("*.*")))

        # Move files
        moved = 0
        for img_file in session_dir.glob("*.jpg"):
            new_name = f"{category}_{existing + moved + 1:04d}.jpg"
            dest = target_dir / new_name
            shutil.move(str(img_file), str(dest))
            moved += 1

        # Clean up temp directory
        shutil.rmtree(session_dir)

        print(f"\n✅ Added {moved} images to {target_dir}")
        print(f"   Total {category} images now: {existing + moved}")

    def view_images(self, session_dir):
        """Quick slideshow of captured images"""
        images = list(session_dir.glob("*.jpg"))

        if not images:
            print("No images found")
            return

        print(f"\n👀 Viewing {len(images)} images")
        print("   Press SPACE for next, Q to return to menu")

        idx = 0
        while idx < len(images):
            img = cv2.imread(str(images[idx]))

            # Add image info
            cv2.putText(img, f"Image {idx+1}/{len(images)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, "SPACE=next, Q=quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Review Images', img)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                idx += 1

        cv2.destroyAllWindows()

    def show_stats(self):
        """Show current dataset statistics"""
        print("\n📊 Current Dataset:")

        if self.data_dir.exists():
            train_hand = len(list((self.data_dir / "train/hand").glob("*.*"))) if (self.data_dir / "train/hand").exists() else 0
            train_not = len(list((self.data_dir / "train/not_hand").glob("*.*"))) if (self.data_dir / "train/not_hand").exists() else 0

            print(f"   Training hands: {train_hand}")
            print(f"   Training not_hands: {train_not}")
            print(f"   Total: {train_hand + train_not}")

        # Check temp folder
        if self.temp_dir.exists():
            temp_sessions = list(self.temp_dir.iterdir())
            if temp_sessions:
                print(f"\n📁 Temp sessions pending review: {len(temp_sessions)}")
                for session in temp_sessions:
                    img_count = len(list(session.glob("*.jpg")))
                    print(f"   - {session.name}: {img_count} images")

def main():
    capture = CaptureWithReview()

    while True:
        print("\n" + "="*60)
        print("🖐️  Hand Detection - Capture & Review")
        print("="*60)

        capture.show_stats()

        print("\n📸 Capture Options:")
        print("  1. Capture HAND images")
        print("  2. Capture NOT_HAND images")
        print("  3. Quick capture both (25 each)")
        print("  4. Review temp folders")
        print("  5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            count = input("How many hand images? [50]: ").strip()
            count = int(count) if count else 50
            session = capture.capture_session("hand", count)
            capture.review_session(session)

        elif choice == '2':
            count = input("How many not_hand images? [50]: ").strip()
            count = int(count) if count else 50
            session = capture.capture_session("not_hand", count)
            capture.review_session(session)

        elif choice == '3':
            print("\n🚀 Quick capture - hands first")
            session = capture.capture_session("hand", 25)
            capture.review_session(session)

            print("\n🚀 Now capture without hands")
            import time
            time.sleep(2)
            session = capture.capture_session("not_hand", 25)
            capture.review_session(session)

        elif choice == '4':
            # Review existing temp folders
            if capture.temp_dir.exists():
                temp_sessions = list(capture.temp_dir.iterdir())
                if temp_sessions:
                    print("\n📁 Temp sessions:")
                    for i, session in enumerate(temp_sessions, 1):
                        img_count = len(list(session.glob("*.jpg")))
                        print(f"  {i}. {session.name} ({img_count} images)")

                    sel = input("\nSelect session to review: ").strip()
                    try:
                        idx = int(sel) - 1
                        if 0 <= idx < len(temp_sessions):
                            session = temp_sessions[idx]
                            img_count = len(list(session.glob("*.jpg")))
                            # Determine category from folder name
                            if "hand" in session.name and "not_hand" not in session.name:
                                category = "hand"
                            elif "not_hand" in session.name:
                                category = "not_hand"
                            else:
                                category = input("Category (hand/not_hand): ").strip()

                            capture.review_session((session, category, img_count))
                    except:
                        print("❌ Invalid selection")
                else:
                    print("No temp sessions found")
            else:
                print("No temp folder exists")

        elif choice == '5':
            print("\n👋 Goodbye!")

            # Check for temp files
            if capture.temp_dir.exists():
                temp_sessions = list(capture.temp_dir.iterdir())
                if temp_sessions:
                    print(f"\n⚠️  You have {len(temp_sessions)} temp sessions")
                    print("   They will be kept for next time")
            break
        else:
            print("❌ Invalid choice")

    print("\n✅ Done!")
    print("\nNext steps:")
    print("  1. python3 train_hands_only.py - Retrain model")
    print("  2. python3 live_demo.py - Test with webcam")

if __name__ == "__main__":
    main()