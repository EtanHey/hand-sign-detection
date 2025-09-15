#!/usr/bin/env python3
"""
Combined workflow: Capture videos and prepare dataset in one step.
This is the recommended tool for workshop participants.
"""
import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        return False
    return True


def main():
    """Main workflow combining video capture and frame extraction."""
    print("=== Hand Classification Dataset Creator ===")
    print("This tool will:")
    print("1. Record two 60-second videos (hands and no hands)")
    print("2. Extract frames at 5 fps (300 images each)")
    print("3. Create/expand a balanced dataset")
    print("4. Split 80/20 for training/validation")
    
    # Check for existing dataset
    existing_count = 0
    if os.path.exists('hand_cls'):
        for split in ['train', 'val']:
            for class_name in ['hand', 'not_hand']:
                path = f'hand_cls/{split}/{class_name}'
                if os.path.exists(path):
                    count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    existing_count += count
    
    if existing_count > 0:
        print(f"\n📊 Found existing dataset with {existing_count} total images")
        print("This is great for adding variety to improve model accuracy!")
    
    # Check if dataset structure exists
    if not os.path.exists('hand_cls'):
        print("\nCreating dataset structure...")
        if not run_command("python3 create_dataset_structure.py", "Creating directories"):
            return 1
    
    # Step 1: Capture videos
    print("\n" + "="*50)
    print("STEP 1: VIDEO CAPTURE")
    print("="*50)
    
    if not run_command("python3 capture_dataset_videos.py", "Capturing videos"):
        print("\nVideo capture failed or was cancelled.")
        return 1
    
    # Check if videos were created
    if not os.path.exists("hand_video.mp4") or not os.path.exists("not_hand_video.mp4"):
        print("\nError: Videos were not created. Exiting.")
        return 1
    
    # Step 2: Extract frames
    print("\n" + "="*50)
    print("STEP 2: FRAME EXTRACTION")
    print("="*50)
    
    if not run_command("python3 extract_frames_to_dataset.py", "Extracting frames"):
        print("\nFrame extraction failed.")
        return 1
    
    # Optional: Clean up video files
    print("\n" + "="*50)
    print("CLEANUP")
    print("="*50)
    
    response = input("\nDelete original video files to save space? (y/n): ").lower()
    if response == 'y':
        try:
            os.remove("hand_video.mp4")
            os.remove("not_hand_video.mp4")
            print("✓ Video files deleted")
        except Exception as e:
            print(f"Warning: Could not delete video files: {e}")
    
    # Final message
    print("\n" + "="*50)
    print("🎉 DATASET READY!")
    print("="*50)
    print("\nYour dataset is now ready for training!")
    print("\nWhat's next:")
    print("1. Upload to RunPod: Follow instructions in README.md")
    print("2. Train locally: yolo classify train model=yolov8n-cls.pt data=hand_cls epochs=15")
    print("3. Run demo: python3 live_demo.py --weights best.pt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())