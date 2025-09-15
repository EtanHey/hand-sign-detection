#!/usr/bin/env python3
"""
Enhanced video capture for hand detection dataset.
Records 20-second videos and extracts more frames per second
to capture odd/transitional hand positions.
"""

import cv2
import time
import os
import sys
import subprocess
from pathlib import Path

def draw_text(frame, text, position=(50, 50), font_scale=1.0, color=(255, 255, 255), thickness=2):
    """Draw text with a dark background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    cv2.rectangle(frame,
                  (position[0] - 10, position[1] - text_height - 10),
                  (position[0] + text_width + 10, position[1] + baseline + 10),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

def countdown_capture(cap, duration=3, video_type="hands"):
    """Show preparation message and countdown before recording starts."""
    # Show preparation message
    prep_start = time.time()
    while time.time() - prep_start < 2.0:
        ret, frame = cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if video_type == "hands":
            draw_text(frame, "GET READY TO SHOW YOUR HANDS!", (50, h//2 - 50), 1.2, (0, 255, 255), 2)
            draw_text(frame, "Move them around, make gestures!", (50, h//2), 1.0, (255, 255, 255), 2)
            draw_text(frame, "Try different angles & positions", (50, h//2 + 50), 0.8, (200, 200, 200), 2)
        else:
            draw_text(frame, "GET READY TO HIDE YOUR HANDS!", (50, h//2 - 50), 1.2, (255, 0, 0), 2)
            draw_text(frame, "Remove all hands from view", (50, h//2), 1.0, (255, 255, 255), 2)
            draw_text(frame, "Move around, show background", (50, h//2 + 50), 0.8, (200, 200, 200), 2)

        cv2.imshow('Video Capture', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            return False

    # Countdown
    for i in range(duration, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ret, frame = cap.read()
            if not ret:
                return False

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Big countdown number
            text = str(i)
            draw_text(frame, text, (w//2 - 50, h//2), 5.0, (0, 255, 255), 8)

            # Instructions
            if video_type == "hands":
                draw_text(frame, "HANDS VISIBLE IN...", (50, 50), 1.0, (0, 255, 0), 2)
            else:
                draw_text(frame, "NO HANDS IN...", (50, 50), 1.0, (255, 0, 0), 2)

            cv2.imshow('Video Capture', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False

    return True

def record_video(cap, output_path, duration=20, video_type="hands"):
    """Record video for specified duration."""
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    frame_count = 0

    print(f"\n📹 Recording {video_type} video ({duration} seconds)...")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)

        # Write original frame
        out.write(frame)
        frame_count += 1

        # Add overlay for display
        remaining = duration - elapsed
        progress = elapsed / duration

        # Progress bar
        bar_width = int(width * 0.8)
        bar_height = 30
        bar_x = int(width * 0.1)
        bar_y = height - 60

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)

        # Instructions
        if video_type == "hands":
            instruction = "SHOW YOUR HANDS! Move them around!"
            color = (0, 255, 0)
        else:
            instruction = "HIDE YOUR HANDS! Move camera around!"
            color = (0, 0, 255)

        draw_text(frame, instruction, (50, 50), 1.3, color, 3)
        draw_text(frame, f"Recording: {int(remaining)}s remaining", (50, 100), 1.0)
        draw_text(frame, f"Frames: {frame_count} @ {fps} fps", (50, 140), 0.8, (200, 200, 200))

        cv2.imshow('Video Capture', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("\nRecording cancelled")
            out.release()
            return False

    out.release()
    print(f"✅ Recorded {frame_count} frames ({frame_count/fps:.1f} seconds)")
    return True

def extract_frames(video_path, output_dir, label, fps_extract=10):
    """Extract frames from video at specified FPS using ffmpeg."""
    print(f"\n🎬 Extracting frames at {fps_extract} fps...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = len(list(Path(output_dir).glob("*.jpg")))

    # Extract frames using ffmpeg
    output_pattern = str(Path(output_dir) / f"{label}_%04d.jpg")

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps_extract}',  # Extract at specified fps
        '-start_number', str(existing + 1),  # Continue numbering
        output_pattern,
        '-loglevel', 'error'
    ]

    try:
        subprocess.run(cmd, check=True)
        # Count new frames
        new_count = len(list(Path(output_dir).glob("*.jpg"))) - existing
        print(f"   ✅ Extracted {new_count} frames to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to extract frames: {e}")
        return False

def main():
    """Main capture and extraction workflow."""
    print("🖐️  Enhanced Hand Detection Data Capture")
    print("="*50)
    print("This will record 20-second videos and extract frames")
    print("at higher FPS to catch transitional hand positions\n")

    # Check for existing data
    data_dir = Path("data/raw")
    if data_dir.exists():
        hand_count = len(list((data_dir / "hand").glob("*.jpg"))) if (data_dir / "hand").exists() else 0
        not_hand_count = len(list((data_dir / "not_hand").glob("*.jpg"))) if (data_dir / "not_hand").exists() else 0

        if hand_count > 0 or not_hand_count > 0:
            print(f"📊 Existing data found:")
            print(f"   Hands: {hand_count} images")
            print(f"   Not hands: {not_hand_count} images")
            print("\n💡 Tips for variety:")
            print("   - Different lighting conditions")
            print("   - Various hand positions and angles")
            print("   - Different distances from camera")
            print("   - Different backgrounds\n")

    print("Press SPACE to begin or ESC to cancel...")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return 1

    # Wait for user
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read from webcam")
            return 1

        frame = cv2.flip(frame, 1)
        draw_text(frame, "Press SPACE to begin recording", (50, 50), 1.0)
        draw_text(frame, "Press ESC to cancel", (50, 100), 0.8, (200, 200, 200))

        cv2.imshow('Video Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            break
        elif key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return 0

    # Record videos
    videos = []

    # 1. Record hands video (20 seconds)
    print("\n📹 Recording 1/2: HANDS")
    if not countdown_capture(cap, 3, "hands"):
        cap.release()
        cv2.destroyAllWindows()
        return 0

    if record_video(cap, "temp_hand_video.mp4", 20, "hands"):
        videos.append(("temp_hand_video.mp4", "hand"))

    # 2. Record no-hands video (20 seconds)
    print("\n📹 Recording 2/2: NO HANDS")
    time.sleep(2)  # Brief pause

    if not countdown_capture(cap, 3, "no_hands"):
        cap.release()
        cv2.destroyAllWindows()
        return 0

    if record_video(cap, "temp_not_hand_video.mp4", 20, "no hands"):
        videos.append(("temp_not_hand_video.mp4", "not_hand"))

    cap.release()
    cv2.destroyAllWindows()

    # Extract frames at higher FPS
    if videos:
        print("\n" + "="*50)
        print("📦 Extracting frames from videos...")

        # Extract at 10 fps (200 frames from 20-second video)
        # This captures more transitional positions
        for video_path, label in videos:
            output_dir = f"data/raw/{label}"
            extract_frames(video_path, output_dir, label, fps_extract=10)

            # Clean up video file
            try:
                os.remove(video_path)
                print(f"   🗑️  Removed temporary video: {video_path}")
            except:
                pass

        print("\n✅ Data capture complete!")

        # Show final counts
        hand_dir = Path("data/raw/hand")
        not_hand_dir = Path("data/raw/not_hand")

        if hand_dir.exists() or not_hand_dir.exists():
            print("\n📊 Total dataset:")
            if hand_dir.exists():
                print(f"   Hands: {len(list(hand_dir.glob('*.jpg')))} images")
            if not_hand_dir.exists():
                print(f"   Not hands: {len(list(not_hand_dir.glob('*.jpg')))} images")

        print("\n🚀 Next steps:")
        print("   1. Review images in data/raw/")
        print("   2. Train with: python3 train_hands_only.py")
        print("   3. Test with: python3 live_demo.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())