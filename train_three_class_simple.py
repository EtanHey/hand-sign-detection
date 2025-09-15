#!/usr/bin/env python3
"""
Simplified three-class training with smart hierarchy
Direct training without complex config
"""

import subprocess
from pathlib import Path
from datetime import datetime

def train_smart_detector():
    """Train with smart hierarchy using direct paths"""

    print("\n🚀 Training Smart Three-Class Detector")
    print("=" * 50)
    print("📝 Detection Priority: Hand > Arm > Nothing")

    # Check data exists
    data_path = Path("data/three_class")
    if not data_path.exists():
        print("❌ Data not found at data/three_class")
        print("   Run first: python3 capture_three_class.py")
        return None

    # Count images
    total = 0
    for split in ['train', 'val']:
        for category in ['hand', 'arm', 'not_hand']:
            path = data_path / split / category
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                total += count
                print(f"{split:5s}/{category:10s}: {count:4d} images")

    print(f"\nTotal: {total} images")

    if total == 0:
        print("\n❌ No images found! Capture data first.")
        return None

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"three_class_smart_{timestamp}"

    # Use absolute path for data
    data_abs = data_path.absolute()

    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8s-cls (better accuracy)")
    print(f"   Data: {data_abs}")
    print(f"   Epochs: 30")
    print(f"   Early stopping: 10 epochs")

    # Build command - simpler without config file
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8s-cls.pt",
        f"data={data_abs}",  # Direct path to data folder
        f"epochs=30",
        f"batch=16",
        f"patience=10",
        f"lr0=0.01",  # Standard learning rate
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
        "plots=True",
        "device=mps" if has_mps() else "device=cpu"
    ]

    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50 + "\n")

    # Run training
    try:
        process = subprocess.run(cmd)

        if process.returncode == 0:
            print("\n✅ Training completed!")

            # Find the best model
            runs_dir = Path("runs/classify") / run_name
            best_model = runs_dir / "weights/best.pt"

            if best_model.exists():
                # Save versioned model
                existing = list(models_dir.glob("three_class_v*.pt"))
                next_version = len(existing) + 1

                dest = models_dir / f"three_class_v{next_version}.pt"
                import shutil
                shutil.copy(best_model, dest)
                print(f"\n📦 Model saved to: {dest}")

                # Update latest
                latest = models_dir / "three_class_detector.pt"
                shutil.copy(best_model, latest)
                print(f"   Latest: {latest}")

                print(f"\n🎥 Test it:")
                print(f"   python3 live_demo_three_class.py")

                return dest
            else:
                print("⚠️  Could not find best.pt")
        else:
            print(f"\n❌ Training failed with code {process.returncode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
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

if __name__ == "__main__":
    model_path = train_smart_detector()
    if model_path:
        print("\n🎉 Smart detector ready!")
        print("The model prioritizes hand detection over arm detection")