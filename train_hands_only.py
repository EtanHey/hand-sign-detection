#!/usr/bin/env python3
"""
Simple hand detection training using existing dataset
Binary classification: hand vs not_hand
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def train_hand_detector():
    """Train YOLO classifier for hand detection using existing data"""

    print("\n🚀 Starting Hand Detection Training")
    print("="*50)

    # Check data exists
    data_path = Path("data/hand_cls")
    if not data_path.exists():
        print("❌ Data not found at data/hand_cls")
        return None

    # Count images
    train_hands = len(list((data_path / "train/hand").glob("*")))
    train_no_hands = len(list((data_path / "train/not_hand").glob("*")))
    val_hands = len(list((data_path / "val/hand").glob("*")))
    val_no_hands = len(list((data_path / "val/not_hand").glob("*")))

    print(f"\n📊 Dataset Statistics:")
    print(f"   Training:   {train_hands:4d} hands, {train_no_hands:4d} not_hands")
    print(f"   Validation: {val_hands:4d} hands, {val_no_hands:4d} not_hands")
    print(f"   Total:      {train_hands + train_no_hands + val_hands + val_no_hands:4d} images")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hand_detect_{timestamp}"

    # Start with small model for quick iteration
    model_size = 'n'  # nano model - fast training
    epochs = 20  # Quick training
    batch = 32  # Good default

    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8{model_size}-cls (classification)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Data path: {data_path}")

    # Build command
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8{model_size}-cls.pt",  # Classification model
        f"data={data_path}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"name={run_name}",
        "patience=5",  # Stop early if no improvement
        "save=True",
        "exist_ok=True",
        "plots=True",  # Generate training plots
        "device=mps" if has_mps() else "device=cpu"
    ]

    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n" + "="*50)
    print("Training started... (this may take a few minutes)")
    print("="*50 + "\n")

    # Run training with YOLO's native output display
    try:
        # Run directly without capturing output - let YOLO handle the display
        process = subprocess.run(cmd)

        if process.returncode == 0:
            print("\n✅ Training completed successfully!")

            # Find the best model
            runs_dir = Path("runs/classify") / run_name
            best_model = runs_dir / "weights/best.pt"

            if best_model.exists():
                # Find next version number
                existing_models = list(models_dir.glob("hand_detector_v*.pt"))
                if existing_models:
                    # Extract version numbers
                    versions = []
                    for model in existing_models:
                        try:
                            version = int(model.stem.split('_v')[1])
                            versions.append(version)
                        except:
                            pass
                    next_version = max(versions) + 1 if versions else 1
                else:
                    next_version = 1

                # Copy to models directory with version
                dest = models_dir / f"hand_detector_v{next_version}.pt"
                import shutil
                shutil.copy(best_model, dest)
                print(f"\n📦 Model saved to: {dest}")

                # Also update the "latest" symlink/copy
                latest = models_dir / "hand_detector.pt"
                shutil.copy(best_model, latest)
                print(f"   Also updated: {latest} (latest)")
                print(f"   Training plots: {runs_dir}")

                print("\n🎥 To see it working LIVE:")
                print("   python3 live_demo.py")
                print("\n   This will open your webcam with real-time detection!")
                print("   - RED = Hand detected (high confidence)")
                print("   - GREEN = Hand detected (lower confidence)")
                print("   - GRAY = No hand")

                return dest
            else:
                print("⚠️  Could not find best.pt model file")
        else:
            print(f"\n❌ Training failed with code {process.returncode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training error: {e}")

    return None

def has_mps():
    """Check if Apple Silicon MPS is available"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

def test_model(model_path):
    """Quick test of the trained model"""
    if not model_path or not model_path.exists():
        print("❌ No model to test")
        return

    print(f"\n🧪 Testing model: {model_path}")

    # Test with a sample image from validation set
    val_hand = Path("data/hand_cls/val/hand")
    if val_hand.exists():
        sample_images = list(val_hand.glob("*"))[:3]

        if sample_images:
            print(f"   Testing on {len(sample_images)} sample images...")

            cmd = [
                "yolo", "classify", "predict",
                f"model={model_path}",
                f"source={sample_images[0]}",
                "save=True"
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print("   ✅ Model inference working!")
                print("   Check runs/classify/predict for results")
            except:
                print("   ⚠️  Test failed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train hand detector")
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only')
    args = parser.parse_args()

    if args.test_only:
        model = Path("models/hand_detector.pt")
        if model.exists():
            test_model(model)
        else:
            print("No model found. Train first!")
    else:
        model_path = train_hand_detector()
        if model_path:
            test_model(model_path)
            print("\n🎉 Hand detector ready!")
            print("\nNext steps:")
            print("1. Test with: python3 train_hands_only.py --test-only")
            print("2. Live demo: python3 live_demo.py")
            print("3. Collect gesture data: python3 collect_data.py")